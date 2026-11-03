"""Use through run_stage.py v2-STAGE; outputs never overwrite version 1."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['preregister','extract','diagnose','develop','calibrate','verify','gate'])
    args=parser.parse_args()
    from wsp.v2 import state
    if args.stage in ['preregister','verify','gate']:
        getattr(state,args.stage)(); print(args.stage+' completed'); return
    if args.stage=='extract':
        from wsp.v2.data import extract
        extract(); return
    from wsp.v2 import calibration
    getattr(calibration,args.stage)()

if __name__=='__main__': main()
