#!/usr/bin/env python3
"""Replay every evaluated arm at the full eval pool, into runs/evalsuite-full/."""
import argparse, json, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, 'scripts')
from score_toolbench import model_spec, NOT_MODELS
REPORTS = Path('runs/evalsuite')
OUT = Path('runs/evalsuite-full')
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kinds', default='catalog')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--status', default='explog/FULLPOOL_STATUS.txt')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    status = Path(args.status)
    for path in sorted(REPORTS.glob('*.json')):
        tag = path.stem
        if tag in NOT_MODELS: continue
        report = json.loads(path.read_text())
        if report.get('kind') not in set(args.kinds.split(',')): continue
        out = OUT / path.name
        if out.exists(): continue
        spec = model_spec(report)
        if spec is None:
            with status.open('a') as h: h.write(f'{tag} skipped (no spec)
')
            continue
        started = time.time()
        r = subprocess.run([sys.executable, 'scripts/eval_suite.py', '--model', spec,
                            '--rows', '999999', '--device', args.device, '--out', str(out)],
                           capture_output=True, text=True)
        Path(f'explog/full_{tag}.log').write_text(r.stdout + r.stderr)
        with status.open('a') as h:
            h.write(f'{tag} rc={r.returncode} secs={time.time()-started:.0f}
')
        print(tag, r.returncode, flush=True)
    with status.open('a') as h: h.write('FULLPOOL-PASS-DONE
')
main()
