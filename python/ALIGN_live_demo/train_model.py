"""Train ALIGN's small, session-separated demonstration classifier.

Inputs are unchanged CSVs produced by collect_csv.py (a directory or ZIP).
Only the Python standard library and the adjacent common.py are required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from common import CHANNELS, WindowBuffer, fit_model, parse_sample, predict


EXPECTED_FIELDS = ["session_id", "subject_id", "label", "seq", "time_us"] + list(CHANNELS)
MAX_CSV_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024


def input_csvs(source: Path):
    """Yield display name and bytes; ZIP entries are never extracted to disk."""
    if source.is_dir():
        # Excel's ~$ files contain lock metadata, not sensor observations.
        files = sorted(p for p in source.rglob("*.csv")
                       if p.is_file() and not p.name.startswith("~$"))
        if not files:
            raise ValueError(f"CSV 파일이 없습니다: {source}")
        total = 0
        for path in files:
            size = path.stat().st_size
            total += size
            if size > MAX_CSV_BYTES or total > MAX_TOTAL_BYTES:
                raise ValueError("학습 입력이 너무 큽니다. 파일당 128MiB, 전체 1GiB 이내로 나눠 주세요.")
            yield str(path.relative_to(source)), path.read_bytes()
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            entries = sorted((x for x in archive.infolist()
                              if not x.is_dir() and x.filename.lower().endswith(".csv")
                              and not Path(x.filename).name.startswith("~$")),
                             key=lambda x: x.filename)
            if not entries:
                raise ValueError("ZIP 안에 CSV 파일이 없습니다.")
            if len({x.filename for x in entries}) != len(entries):
                raise ValueError("ZIP에 같은 경로의 CSV 항목이 중복되어 있습니다.")
            if (any(x.file_size > MAX_CSV_BYTES for x in entries)
                    or sum(x.file_size for x in entries) > MAX_TOTAL_BYTES):
                raise ValueError("학습 입력이 너무 큽니다. 파일당 128MiB, 전체 1GiB 이내로 나눠 주세요.")
            for entry in entries:
                yield entry.filename, archive.read(entry)
    else:
        raise ValueError("--data에는 수집 CSV가 있는 폴더 또는 ZIP 파일을 지정하세요.")


def read_session(name: str, content: bytes):
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: UTF-8 CSV가 아닙니다.") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    if reader.fieldnames != EXPECTED_FIELDS:
        raise ValueError(f"{name}: 수집 CSV 열 이름/순서가 다릅니다. 원본 collect_csv.py CSV만 넣어 주세요.")
    buffer = WindowBuffer()
    session_id = subject_id = label = None
    last_seq = last_time = first_time = None
    features = []
    source_window_ends = []
    rows = gaps = timing_anomalies = near_scale = interruptions = 0
    signal_hash = hashlib.sha256()
    try:
        for row in reader:
            line = reader.line_num
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{name}:{line}: 열 개수가 맞지 않습니다.")
            sid, subject, raw_label = row["session_id"], row["subject_id"], row["label"]
            if not sid.strip() or not subject.strip() or raw_label not in ("0", "1"):
                raise ValueError(f"{name}:{line}: 세션/피험자 ID 또는 라벨(0/1)이 잘못되었습니다.")
            if session_id is None:
                session_id, subject_id, label = sid, subject, int(raw_label)
            elif (sid, subject, int(raw_label)) != (session_id, subject_id, label):
                raise ValueError(f"{name}:{line}: 한 CSV 안에서 세션/피험자/라벨이 바뀌었습니다.")
            try:
                seq, timestamp, values = parse_sample(",".join(row[k] for k in EXPECTED_FIELDS[3:]))
            except ValueError as exc:
                raise ValueError(f"{name}:{line}: 잘못된 센서 행: {exc}") from exc
            if last_seq is not None:
                if seq <= last_seq or timestamp <= last_time:
                    raise ValueError(f"{name}:{line}: 순번/시간 중복 또는 역행이 있습니다.")
                steps = seq - last_seq
                gaps += steps - 1
                timing_anomalies += int(abs(timestamp - last_time - steps * 20000) > 5000)
            else:
                gaps += seq
                first_time = timestamp
            last_seq, last_time = seq, timestamp
            rows += 1
            near_scale += int(any(abs(v) >= (1.98 if i % 6 < 3 else 247.5)
                                  for i, v in enumerate(values)))
            # Ignore metadata and time for this fingerprint: relabeling or copying
            # a recording under a fresh session ID must not leak across folds.
            signal_hash.update(json.dumps(values, separators=(",", ":"), allow_nan=False).encode("ascii"))
            signal_hash.update(b"\n")
            feature, status = buffer.feed(seq, timestamp, values)
            interruptions += int(status in ("gap", "saturated"))
            if feature is not None:
                features.append(feature)
                source_window_ends.append({"seq": seq, "time_us": timestamp})
    except csv.Error as exc:
        raise ValueError(f"{name}:{reader.line_num}: CSV 문법 오류: {exc}") from exc
    if not rows:
        raise ValueError(f"{name}: 데이터 행이 없습니다.")
    if not features:
        raise ValueError(f"{name}: 연속된 유효 50행 구간이 없습니다. 누락·시간 간격·포화를 확인하세요.")
    return {
        "file": name, "sha256": hashlib.sha256(content).hexdigest(),
        "signal_sha256": signal_hash.hexdigest(),
        "session_id": session_id, "subject_id": subject_id, "label": label,
        "rows": rows, "usable_windows": len(features),
        "unused_rows": rows - len(features) * 50,
        "missing_slots": gaps, "timing_anomalies": timing_anomalies,
        "near_full_scale_rows": near_scale, "window_interruptions": interruptions,
        "received_hz": (rows - 1) * 1_000_000 / (last_time - first_time) if rows > 1 else None,
        "features": features, "window_ends": source_window_ends,
    }


def load_sessions(source: Path):
    sessions = []
    seen_ids, seen_signals, seen_files = {}, {}, {}
    for name, data in input_csvs(source):
        session = read_session(name, data)
        for key, seen, message in (
                ("session_id", seen_ids, "세션 ID가 중복"),
                ("sha256", seen_files, "동일한 CSV 파일이 중복"),
                ("signal_sha256", seen_signals, "같은 센서 기록이 다른 파일에 중복")):
            value = session[key]
            if value in seen:
                raise ValueError(f"{message}입니다: {seen[value]} / {name}. 중복 사본을 제외하세요.")
            seen[value] = name
        sessions.append(session)
    return sorted(sessions, key=lambda s: (s["session_id"], s["file"]))


def make_folds(sessions, split):
    """Choose group splits without using features or prediction results."""
    if split == "session":
        groups = [[i for i, s in enumerate(sessions) if s["label"] == c] for c in (0, 1)]
        if min(map(len, groups)) < 2:
            raise ValueError("세션 분리 평가에는 라벨 0과 1 각각 최소 2개의 독립 세션이 필요합니다.")
        count = min(5, len(groups[0]), len(groups[1]))
        test_groups = [sorted(groups[0][fold::count] + groups[1][fold::count]) for fold in range(count)]
    else:
        subjects = sorted({s["subject_id"] for s in sessions})
        if len(subjects) < 2:
            raise ValueError("피험자 분리 평가에는 서로 다른 subject_id가 최소 2개 필요합니다.")
        test_groups = [[i for i, s in enumerate(sessions) if s["subject_id"] == subject]
                       for subject in subjects]
    folds = []
    for test in test_groups:
        train = [i for i in range(len(sessions)) if i not in test]
        if {sessions[i]["label"] for i in train} != {0, 1}:
            raise ValueError("일부 평가 회차의 학습 집합에 라벨 0/1이 모두 없습니다. 각 피험자/세션의 양쪽 라벨을 추가하세요.")
        folds.append((train, test))
    return folds


def metrics(predictions):
    matrix = [[0, 0], [0, 0]]
    for row in predictions:
        matrix[row["true_label"]][row["predicted_label"]] += 1
    total = sum(map(sum, matrix))
    recall = [matrix[c][c] / sum(matrix[c]) if sum(matrix[c]) else None for c in (0, 1)]
    present = [x for x in recall if x is not None]
    return {"windows": total, "correct": matrix[0][0] + matrix[1][1],
            "accuracy": (matrix[0][0] + matrix[1][1]) / total if total else None,
            "balanced_accuracy": sum(present) / len(present) if present else None,
            "recall_label0": recall[0], "recall_label1": recall[1],
            "confusion_matrix_true_rows_predicted_columns": matrix}


def flatten(sessions, indices):
    features, labels = [], []
    for i in indices:
        features.extend(sessions[i]["features"])
        labels.extend([sessions[i]["label"]] * len(sessions[i]["features"]))
    return features, labels


def evaluate(sessions, split):
    predictions, fold_reports = [], []
    for fold, (train, test) in enumerate(make_folds(sessions, split), 1):
        features, labels = flatten(sessions, train)
        params = fit_model(features, labels)
        fold_predictions = []
        for index in test:
            session = sessions[index]
            for wi, feature in enumerate(session["features"]):
                result = predict(feature, params)
                row = {"fold": fold, "session_id": session["session_id"],
                       "subject_id": session["subject_id"], "window_index": wi,
                       "source_window_end": session["window_ends"][wi],
                       "true_label": session["label"], "predicted_label": result["label"]}
                fold_predictions.append(row)
        predictions.extend(fold_predictions)
        fold_reports.append({"fold": fold,
                             "train_sessions": [sessions[i]["session_id"] for i in train],
                             "test_sessions": [sessions[i]["session_id"] for i in test],
                             "train_subjects": sorted({sessions[i]["subject_id"] for i in train}),
                             "test_subjects": sorted({sessions[i]["subject_id"] for i in test}),
                             "train_windows": len(features), **metrics(fold_predictions)})
    session_votes = []
    for session in sessions:
        rows = [p for p in predictions if p["session_id"] == session["session_id"]]
        n1 = sum(p["predicted_label"] for p in rows)
        vote = int(n1 * 2 > len(rows)) if n1 * 2 != len(rows) else None
        session_votes.append({"session_id": session["session_id"], "true_label": session["label"],
                              "predicted_label": vote, "tie": vote is None,
                              "windows": len(rows), "votes_label1": n1})
    return {"split": split, "fold_count": len(fold_reports), **metrics(predictions),
            "session_votes_correct": sum(x["predicted_label"] == x["true_label"] for x in session_votes),
            "session_votes_total": len(session_votes),
            "session_votes_tied": sum(x["tie"] for x in session_votes),
            "session_votes": session_votes, "folds": fold_reports, "predictions": predictions,
            "notes": ["같은 세션은 학습과 평가에 섞이지 않습니다.",
                      "표준화 평균/표준편차는 각 회차의 학습 데이터에서만 계산했습니다.",
                      "최종 저장 모델은 평가 후 전체 입력 세션으로 다시 학습합니다. 위 점수는 그 모델의 독립 테스트 점수가 아닙니다.",
                      "session 분리는 새로운 피험자에 대한 성능을 증명하지 않습니다."]}


def run(args):
    source, out = Path(args.data).expanduser().resolve(), Path(args.out).expanduser().resolve()
    if not args.overwrite:
        exists = [p for p in (out / "model.json", out / "evaluation.json") if p.exists()]
        if exists:
            raise ValueError(f"결과가 이미 있습니다: {exists[0]}. 새 --out 폴더를 사용하세요. 의도한 교체는 --overwrite로 실행합니다.")
    sessions = load_sessions(source)
    evaluation = evaluate(sessions, args.split)
    public_sessions = [{k: v for k, v in s.items() if k not in ("features", "window_ends")} for s in sessions]
    evaluation["input_sessions"] = public_sessions
    all_features, all_labels = flatten(sessions, range(len(sessions)))
    params = fit_model(all_features, all_labels)
    model = {
        "schema_version": 1, "channels": list(CHANNELS), "window_rows": 50,
        "target_hz": 50, "feature": "channel_means", "method": "standardized_nearest_centroid",
        "kind": args.kind,
        "labels": {"0": "state 0", "1": "state 1"} if args.kind == "hand" else {"0": "normal", "1": "poor"},
        "params": params, "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_sessions": [s["session_id"] for s in sessions],
        "training_subjects": sorted({s["subject_id"] for s in sessions}),
        "training_files": public_sessions, "training_windows": len(all_features),
        "evaluation": {k: v for k, v in evaluation.items()
                       if k not in ("input_sessions", "predictions", "folds", "session_votes")},
        "units": {"acceleration": "g", "gyroscope": "deg/s"},
        "scope": "hand-tilted sensor demonstration" if args.kind == "hand" else "body-worn experimental classifier",
    }
    out.mkdir(parents=True, exist_ok=True)
    for filename, document in (("evaluation.json", evaluation), ("model.json", model)):
        with (out / filename).open("w" if args.overwrite else "x", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write("\n")
    print(f"학습 완료: {len(sessions)}세션, {len(all_features)}개의 약 1초 구간")
    print(f"세션/피험자 분리 평가: {evaluation['correct']}/{evaluation['windows']} ({evaluation['accuracy']:.1%})")
    print(f"모델 저장: {out / 'model.json'}")
    print(f"평가 기록: {out / 'evaluation.json'}")
    if args.kind == "hand":
        print("현재 모델은 손으로 기울인 센서 상태 구분용입니다. 실제 자세 성능을 의미하지 않습니다.")
    else:
        print("착용 데이터로 학습했습니다. 새로운 세션/날짜/사람의 독립 평가를 추가하세요.")
    print("주의: CSV 자체만으로 손 기울기/착용 데이터를 자동 구별할 수 없습니다. --kind에 맞는 폴더만 사용하세요.")
    ignored = sum(s["unused_rows"] for s in sessions)
    print(f"구간에 포함되지 않은 행: {ignored} (끝의 남은 행, 누락/시간 이상/포화에 따른 중단 포함)")
    return model, evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="ALIGN CSV로 모델 학습 및 그룹 분리 평가")
    parser.add_argument("--data", required=True, help="원본 CSV 폴더 또는 ZIP")
    parser.add_argument("--kind", choices=("hand", "body"), required=True,
                        help="hand: 손 기울기 연습 / body: 실제 착용 데이터")
    parser.add_argument("--split", choices=("session", "subject"), default="session")
    parser.add_argument("--out", required=True, help="새 결과 폴더")
    parser.add_argument("--overwrite", action="store_true", help="기존 model/evaluation 교체 허용")
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (ValueError, OSError, csv.Error, zipfile.BadZipFile, RuntimeError) as exc:
        print(f"학습을 완료하지 못했습니다: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
