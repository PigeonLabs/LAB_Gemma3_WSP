"""Capture a stage, refresh reports, and queue/publish even failed experiments."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

from publication import ROOT, queue, publish_git, utcnow


def run(stage, command, root=ROOT, transport='connector'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', stage):
        raise ValueError('Stage name must be alphanumeric, underscore or hyphen')
    out = root/'artifacts/stages'
    out.mkdir(parents=True, exist_ok=True)
    lockdir = root/'.publication'; lockdir.mkdir(exist_ok=True)
    with (lockdir/'stage.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ident = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+stage
        log = out/(ident+'.log')
        record = {'stage': stage, 'command': command, 'started_at': utcnow(),
                  'log': str(log.relative_to(root)), 'status': 'running',
                  'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for d in ['scripts', 'wsp', 'plans', 'tests'] for p in (root/d).rglob('*')
                                    if p.is_file() and '__pycache__' not in str(p)}}
        child = None
        interrupted = []
        def stop(sig, frame):
            interrupted.append(sig)
            if child is not None and child.poll() is None:
                os.killpg(child.pid, sig)
        old = {s: signal.signal(s, stop) for s in [signal.SIGTERM, signal.SIGINT]}
        rc = 1
        try:
            with log.open('x') as stream:
                stream.write(json.dumps({'command': command, 'started_at': record['started_at']})+'\n'); stream.flush()
                env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH', ''))
                child = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                         env=env, start_new_session=True)
                rc = child.wait()
        except Exception as exc:
            with log.open('a') as stream:
                stream.write(f'runner_error: {type(exc).__name__}: {exc}\n')
        finally:
            for s, handler in old.items(): signal.signal(s, handler)
            record.update(ended_at=utcnow(), exit_code=rc,
                          status='interrupted' if interrupted else 'complete' if rc == 0 else 'failed',
                          signals=interrupted)
            (out/(ident+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        errors = []
        # Version 2 must never regenerate the frozen version-1 evidence.
        reports = ['report_v2.py'] if stage.startswith('v2-') else ['summarize.py', 'build_readme.py']
        for name in reports:
            script = root/'scripts'/name
            if script.exists():
                try:
                    subprocess.run([sys.executable, str(script)], cwd=root, check=True, capture_output=True)
                except subprocess.CalledProcessError as exc:
                    errors.append({'script': name, 'error': exc.stderr.decode(errors='replace')})
        if errors:
            record['report_errors'] = errors
            (out/(ident+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        folder = queue(stage, root)
        print(json.dumps({'stage_status': record['status'], 'log': str(log),
                          'publication': 'pending', 'snapshot': str(folder)}), flush=True)
        if transport == 'git':
            try:
                print(json.dumps(publish_git(folder, root)), flush=True)
            except Exception as exc:
                print(f'Publication pending; retry snapshot after resolving transport: {type(exc).__name__}', file=sys.stderr)
                return rc or 2
        return rc if rc else (3 if errors else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage')
    parser.add_argument('--publish', choices=['connector', 'git'], default='connector')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('A command after -- is required')
    raise SystemExit(run(args.stage, command, transport=args.publish))


if __name__ == '__main__': main()
