"""Finalize CPU-only diagnostic reporting; never launch the next GPU stage."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wsp.v2.state import ROOT,PUBLIC,verify,file_hash,atomic_json
from audit_v2_diagnostics import write_report
from plot_v2_diagnostics import build

def main():
    verify()
    audit=json.loads((PUBLIC/'diagnostic_audit.json').read_text())
    for name,sha in audit['raw_sha256'].items(): assert file_hash(ROOT/name)==sha,name
    assert audit['episodes']==640 and all(audit['unrun_stage_files_absent'].values())
    for name in audit['unrun_stage_files_absent']: assert not (PUBLIC/name).exists(),name
    write_report(audit);build()
    atomic_json(PUBLIC/'final_report_checks.json',{'raw_episodes':640,'raw_hashes_unchanged':True,
        'v1_and_v2_frozen_sources_unchanged':True,'development_and_independent_calibration_not_run':True,
        'thresholds_unchanged':True,'recurring_automation_created':False,
        'posthoc_non_tie_sensitivity_only':True},immutable=True)
    print('Final report generated; raw data and frozen sources unchanged; no further model inference.')

if __name__=='__main__':main()
