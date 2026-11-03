# Implementation and observed stopping outcome

The v1 code path is implemented, with a frozen scientific configuration and source
hashes. **The confirmatory attack experiment did not run**: the actual observer
failed the preregistered independent baseline gate.

| Component | Implementation | Real execution in v1 |
|---|---|---|
| Preregistration/source lock | complete | complete |
| Static real-weight circuit extraction | complete | 272 circuits |
| W/S/M/R capability-separated policy and norm budget | complete; unit-tested | no final-test candidate selection after failed gate |
| Development format selection | complete | 288 A/B scores |
| Independent calibration | complete | 64 A/B scores from 32 circuits |
| Full-model BF16/FP32 preservation | implemented | not run in v1; earlier pilot is separate |
| Held-out binary/prose evaluation | implemented | not run |
| Layer-cluster analysis and simultaneous intervals | implemented; synthetic fixture tested | no real attack outcomes |
| Subset reversal and defenses | implemented | not run |
| Conditional semantic extension | implemented | not run |
| Chunk journals, resumption, publication, reports | complete | exercised |

## Measured result

- Development table: 59/96 (61.46%).
- Development equations: 61/96 (63.54%), selected by the frozen rule.
- Development grouped arrays: 57/96 (59.38%).
- Independent calibration: **33/64 (51.56%)**; **2 UNKNOWN**; required **80%**.
- The 64 calibration responses are two wordings of 32 circuits, not 64 independent
  target models. Truth labels are balanced (32 A, 32 B).
- Median local A/B scoring latency: **0.393 s**; p90 **0.488 s**; observer load **6.95 s**.
- The 9,984-call main scoring projection is **1.09 h** using the median and **1.35 h**
  using p90. This is not a confidence interval or an end-to-end estimate. It excludes
  journal IO, preservation, prose, loading and retries. Main scoring was not run.

Raw prompts, probabilities, labels and timing are in
[`artifacts/study_v1/calibration/`](../artifacts/study_v1/calibration/).
The machine-readable decision is
[`calibration_summary.json`](../artifacts/study_v1/calibration_summary.json).
The [report](../artifacts/study_v1/REPORT.md) keeps unrun figures explicitly marked.

## What this does and does not establish

The result establishes that this local observer / 16-unit projection / no-tool
format did not meet the prespecified baseline. It does **not** establish absence
of a weight-information advantage, successful explanation manipulation, or a
lying neuron. Changing the observer, giving it calculation tools, or simplifying
the circuit would be a new protocol; it is not silently substituted into v1.

Downstream real-GPU paths remain **unvalidated on the full planned workload**.
Unit tests and synthetic fixtures are not a substitute for those conditional
experiments. The original 12-prompt preservation pilot remains separate evidence.
No paid API or background unattended uploader was configured.
