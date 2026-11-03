import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from publication import queue, acknowledge
from run_stage import run


class PublicationTests(unittest.TestCase):
    def test_snapshot_excludes_models_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'models').mkdir(); (root/'models/secret.json').write_text('excluded')
            (root/'README.md').write_text('before')
            snapshot = queue('test', root)
            m = json.loads((snapshot/'manifest.json').read_text())
            self.assertEqual([e['path'] for e in m['files']], ['README.md'])
            (root/'README.md').write_text('after')
            self.assertEqual((snapshot/'objects'/m['files'][0]['sha']).read_text(), 'before')
            bad = {'tree': [{'path': 'README.md', 'type': 'blob', 'sha': 'wrong'}]}
            with self.assertRaises(ValueError): acknowledge(snapshot, 'commit', bad)
            self.assertFalse((snapshot/'receipt.json').exists())
            good = {'tree': [{'path': 'README.md', 'type': 'blob', 'sha': m['files'][0]['sha']}]}
            self.assertEqual(acknowledge(snapshot, 'commit', good)['status'], 'published')

    def test_rejects_credential_and_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'README.md').write_text('gh'+'p_'+'x'*36)
            with self.assertRaises(ValueError): queue('secret', root)
            (root/'README.md').unlink()
            (root/'README.md').symlink_to('/etc/hostname')
            with self.assertRaises(ValueError): queue('symlink', root)

    def test_success_and_failure_both_preserve_logs_and_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for code, expected in [(0, 'complete'), (7, 'failed')]:
                result = run('test-'+str(code), [sys.executable, '-c',
                             f'import sys; print("stage-output"); sys.exit({code})'], root)
                self.assertEqual(result, code)
                record = next(p for p in (root/'artifacts/stages').glob('*.json')
                              if json.loads(p.read_text())['stage'] == 'test-'+str(code))
                metadata = json.loads(record.read_text())
                self.assertEqual(metadata['status'], expected)
                self.assertIn('stage-output', (root/metadata['log']).read_text())
            self.assertEqual(len(list((root/'.publication').glob('*/manifest.json'))), 2)


if __name__ == '__main__': unittest.main()
