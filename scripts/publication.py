"""Allowlisted, content-addressed publication snapshots and verified receipts."""
import argparse
import base64
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'PigeonLabs/LAB_Gemma3_WSP'
TOP_FILES = {'.gitignore', 'AGENTS.md', 'README.md', 'RESULTS.md'}
FOLDERS = {'scripts', 'wsp', 'tests', 'plans', 'docs', 'artifacts'}
EXTENSIONS = {'.py', '.md', '.json', '.jsonl', '.log', '.svg', '.safetensors'}
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|sk-proj-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def git_blob(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def files(root=ROOT):
    result = []
    for p in sorted(root.rglob('*')):
        relative = p.relative_to(root)
        # Models and queues are deliberately never opened/read here.
        if relative.parts[0] not in FOLDERS and str(relative) not in TOP_FILES:
            continue
        if '__pycache__' in relative.parts or not p.is_file():
            continue
        if str(relative) not in TOP_FILES and p.suffix not in EXTENSIONS:
            continue
        if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
            raise ValueError(f'Unsafe publication path: {relative}')
        if any(part.startswith('.') for part in relative.parts) and str(relative) != '.gitignore':
            continue
        if p.stat().st_size > 10_000_000:
            raise ValueError(f'File exceeds publication limit: {relative}')
        data = p.read_bytes()
        if SECRET.search(data):
            raise ValueError(f'Potential credential in {relative}; publication stopped')
        result.append((str(relative), data))
    return result


def queue(label, root=ROOT):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    ident = stamp+'-'+re.sub('[^A-Za-z0-9_-]', '-', label)[:80]
    folder = root / '.publication' / ident
    objects = folder / 'objects'
    objects.mkdir(parents=True)
    entries = []
    for path, data in files(root):
        sha = git_blob(data)
        (objects/sha).write_bytes(data)
        entries.append({'path': path, 'sha': sha, 'sha256': hashlib.sha256(data).hexdigest(),
                        'size': len(data), 'mode': '100644', 'type': 'blob'})
    manifest = {'id': ident, 'repository': REPOSITORY, 'branch': 'main',
                'label': label, 'created_at': utcnow(), 'status': 'pending', 'files': entries}
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return folder


def acknowledge(folder, commit, remote_tree):
    manifest = json.loads((folder/'manifest.json').read_text())
    if remote_tree.get('truncated'):
        raise ValueError('Cannot verify a truncated remote tree')
    remote = {e['path']: e['sha'] for e in remote_tree['tree'] if e['type'] == 'blob'}
    for entry in manifest['files']:
        if remote.get(entry['path']) != entry['sha']:
            raise ValueError(f"Remote blob mismatch: {entry['path']}")
    receipt = {'status': 'published', 'commit': commit, 'verified_at': utcnow(),
               'file_count': len(manifest['files']),
               'url': f'https://github.com/{REPOSITORY}/commit/{commit}'}
    (folder/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def publish_git(folder, root=ROOT):
    """Optional unattended transport, only with existing normal Git credentials."""
    import os
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0')
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, env=env, text=True, stderr=subprocess.STDOUT).strip()
    remote = git('remote', 'get-url', 'origin')
    if remote not in (f'https://github.com/{REPOSITORY}.git', f'git@github.com:{REPOSITORY}.git'):
        raise ValueError('Unexpected Git origin')
    if git('branch', '--show-current') != 'main':
        raise ValueError('Expected main branch')
    if git('diff', '--cached', '--name-only'):
        raise ValueError('Pre-existing staged changes; refusing to include them')
    manifest = json.loads((folder/'manifest.json').read_text())
    for entry in manifest['files']:
        if git_blob((root/entry['path']).read_bytes()) != entry['sha']:
            raise ValueError('Working files changed after snapshot')
    git('add', '--', *[e['path'] for e in manifest['files']])
    if git('diff', '--cached', '--name-only'):
        git('commit', '-m', 'experiment: '+manifest['label'])
    git('push', 'origin', 'HEAD:main')  # Normal fast-forward only.
    commit = git('rev-parse', 'HEAD')
    if git('ls-remote', 'origin', 'refs/heads/main').split()[0] != commit:
        raise ValueError('Remote HEAD verification failed')
    tree = {'tree': []}
    for entry in manifest['files']:
        sha = git('rev-parse', f"{commit}:{entry['path']}")
        tree['tree'].append({'path': entry['path'], 'type': 'blob', 'sha': sha})
    return acknowledge(folder, commit, tree)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    q = sub.add_parser('queue'); q.add_argument('label')
    sub.add_parser('pending')
    e = sub.add_parser('export'); e.add_argument('folder')
    a = sub.add_parser('ack'); a.add_argument('folder'); a.add_argument('--commit', required=True); a.add_argument('--tree', required=True)
    g = sub.add_parser('git'); g.add_argument('folder')
    args = parser.parse_args()
    if args.command == 'queue':
        print(queue(args.label))
    elif args.command == 'pending':
        print(json.dumps([str(p.parent) for p in sorted((ROOT/'.publication').glob('*/manifest.json')) if not (p.parent/'receipt.json').exists()]))
    elif args.command == 'export':
        folder = Path(args.folder)
        m = json.loads((folder/'manifest.json').read_text())
        for e in m['files']:
            data = (folder/'objects'/e['sha']).read_bytes()
            if git_blob(data) != e['sha']:
                raise ValueError('Corrupt snapshot')
            e['encoding'] = 'base64' if e['path'].endswith('.safetensors') else 'utf-8'
            e['content'] = base64.b64encode(data).decode() if e['encoding'] == 'base64' else data.decode('utf-8')
        print(json.dumps(m))
    elif args.command == 'ack':
        print(json.dumps(acknowledge(Path(args.folder), args.commit, json.loads(Path(args.tree).read_text()))))
    else:
        print(json.dumps(publish_git(Path(args.folder))))


if __name__ == '__main__':
    main()
