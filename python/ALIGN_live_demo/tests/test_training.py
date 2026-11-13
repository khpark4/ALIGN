import argparse
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CHANNELS, load_model
from train_model import EXPECTED_FIELDS, evaluate, load_sessions, read_session, run


def csv_bytes(sid, subject, label, offset=0):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(EXPECTED_FIELDS)
    for seq in range(100):
        # Four independent recordings, identical format but distinct signals.
        vals = [.1 + .6 * label + offset + (seq % 7) * .0001] * 12
        writer.writerow([sid, subject, label, seq, 1000000 + seq * 20000] + vals)
    return stream.getvalue().encode('utf-8')


class TrainingTests(unittest.TestCase):
    def build_data(self, folder):
        data = Path(folder) / 'data'
        data.mkdir()
        for i, (subject, label) in enumerate([('P01', 0), ('P01', 1), ('P02', 0), ('P02', 1)]):
            (data / f'{i}.csv').write_bytes(csv_bytes(f'session{i}', subject, label, .005 * i))
        return data

    def test_train_to_saved_model_and_no_silent_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build_data(tmp)
            out = Path(tmp) / 'model'
            args = argparse.Namespace(data=str(data), kind='body', split='session',
                                      out=str(out), overwrite=False)
            with redirect_stdout(io.StringIO()):
                model, evaluation = run(args)
            self.assertEqual(model['kind'], 'body')
            self.assertEqual(load_model(out / 'model.json')['training_windows'], 8)
            self.assertEqual(evaluation['windows'], 8)
            self.assertEqual(evaluation['correct'], 8)
            self.assertEqual(len(model['training_sessions']), 4)
            with self.assertRaises(ValueError):
                run(args)

    def test_both_split_modes_keep_groups_disjoint_and_test_each_session_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            sessions = load_sessions(self.build_data(tmp))
            for split in ['session', 'subject']:
                with self.subTest(split=split):
                    report = evaluate(sessions, split)
                    tested = []
                    for fold in report['folds']:
                        self.assertFalse(set(fold['train_sessions']) & set(fold['test_sessions']))
                        if split == 'subject':
                            self.assertFalse(set(fold['train_subjects']) & set(fold['test_subjects']))
                        tested.extend(fold['test_sessions'])
                    self.assertEqual(sorted(tested), sorted(s['session_id'] for s in sessions))
                    self.assertEqual(report['windows'], 8)

    def test_copied_signal_with_new_id_cannot_leak_into_another_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build_data(tmp)
            (data / 'copy.csv').write_bytes(csv_bytes('new_session', 'P09', 0, 0))
            with self.assertRaisesRegex(ValueError, '중복'):
                load_sessions(data)

    def test_mixed_label_and_invalid_numeric_data_fail_explicitly(self):
        data = csv_bytes('s', 'P01', 0).decode('utf-8')
        mixed = data.replace('s,P01,0,99,', 's,P01,1,99,')
        bad = data.replace(',0.1,', ',nan,', 1)
        for value in [mixed, bad]:
            with self.subTest(data=value[-100:]), self.assertRaises(ValueError):
                read_session('bad.csv', value.encode('utf-8'))


    def test_excel_lock_files_in_directory_are_not_training_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build_data(tmp)
            (data / '~$session0.csv').write_bytes(b'Excel lock metadata, not sensor CSV')
            sessions = load_sessions(data)
            self.assertEqual(len(sessions), 4)
            self.assertEqual(evaluate(sessions, 'session')['windows'], 8)

    def test_excel_lock_files_in_zip_are_not_training_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = self.build_data(tmp)
            archive = Path(tmp) / 'measurements.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for p in data.glob('*.csv'):
                    z.write(p, 'sessions/' + p.name)
                z.writestr('sessions/~$session0.csv', b'Excel lock metadata')
            self.assertEqual(len(load_sessions(archive)), 4)



if __name__ == '__main__':
    unittest.main()
