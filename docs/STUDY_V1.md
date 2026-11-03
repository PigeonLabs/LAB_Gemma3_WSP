# Gemma 3 4B explanation manipulation — preregistered study v1

## Claim and roles

Test whether correctly aligned real weight information increases **targeted false
external explanations**, relative to norm-only and misaligned information, under
function-preserving edits. The target is the verified Gemma 3 4B IT checkpoint;
the observer is the existing local Gemma 4 12B IT checkpoint. The current assistant
authors a frozen static policy. Repeated candidate seeds are **not** independent
frontier-LLM editor calls. No paid API, training, gradient optimization or test-time
observer-feedback search is used. Images and whole-checkpoint understanding are
outside the main claim.

Related work already manipulates explanations while retaining task accuracy:
[Viering et al.](https://arxiv.org/abs/1907.10901),
[Heo et al.](https://arxiv.org/abs/1902.02041).
This experiment combines external weight-reading LLMs, algebraic function
preservation, information controls, held-out evaluation and reversal tests.
It follows [Weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
and distinguishes its claim from general independent-LLM capability.

## Dataset and access separation

| Split | Layers | Circuits |
|---|---|---:|
| Calibration | 0, 11, 17, 33 | 32 |
| Development | 3, 8, 13, 18, 23, 28 | 48 |
| Test | remaining 24 layers | 192 |

Each layer contributes eight disjoint groups of 16 actual neurons. A seeded
permutation partitions all 10,240 neurons into at most 640 candidate groups.
For each group, choose the two axes maximizing summed gate/up squared norms and
the readout coordinate maximizing summed down squared norm. Use input scale 16.
The first eight groups with relative influence margin >=0.2 and total influence
>=1e-8 are accepted. No observer output affects extraction. Reverse axis labels
on alternating accepted groups (four per layer); this balances naming operations,
not necessarily truth prevalence. Report observed truth prevalence separately.

Evaluate all four binary inputs to
`F(A,B)=sum d_i*GELU(gA_i*A+gB_i*B)*(uA_i*A+uB_i*B)`.
Compare the mean absolute changes from toggling each input. Use the actual
tanh-approximate GELU formula. Recheck ground truth after 12-significant-digit
serialization. Do not relax extraction thresholds if the sample is insufficient.
Test circuits and candidate plans stay in `.study_private/` until frozen final
evaluation finishes. This is an agent workflow boundary, not an OS security sandbox.

## Information controls and equal editing budget

Four neurons per circuit: up rows ×8, corresponding down columns ÷8, gates intact.
This changes 20,480 scalar entries in full Gemma, not merely the observer's table.
`U'=SU, D'=DS^-1` preserves the real-valued gated-MLP function.

| Arm | Input to selection code |
|---|---|
| O | no edit |
| W | actual projected g/u/d coefficients |
| S | full-row/full-column gate/up/down norms only |
| M | W coefficients with incorrect identity correspondence |
| R | feasible combination IDs only |

W scores neuron i as `(|u_target|-|u_other|)*hypot(gA,gB)*abs(d)`.
S uses the full norm product. M uses **the same raw-score function as W**, after
deranging rows within four groups sorted by full perturbation energy. R is uniform.
The budget allocator is common infrastructure, outside the policy capability views.
It enumerates 1,820 four-neuron subsets. Each subset norm is
`sqrt(sum(49*up_norm² + (7/8)²*down_norm²))`; the common reference is the median
over all subsets. Retain only norms within ±5% of that reference.
W/S/M sample one of the eight highest score-sum feasible subsets; R samples any
feasible subset. Use seeds 4101, 4102, 4103. No scoring feedback selects candidates.

All trials receive the same preassigned false target (opposite exact truth).
The observer never receives that target, the arm, the plan, or original predictions.
The editor's supplied target does reveal which binary claim is false; this is
intentional attack-goal information, shared across arms, not oracle rollout access.

## Calibration and stopping rule

Three formats are frozen in code: numeric row table, per-unit equations, and
column-grouped arrays. Evaluate two calibration wordings on all 48 development
circuits for each format (288 calls). Select the highest development accuracy,
breaking ties by declaration order. Evaluate **only that selected format** on
32 separate calibration circuits ×2 wordings (64 calls).
Require >=80% accuracy on both development and calibration. UNKNOWN counts wrong.
The absolute A/B completion log-likelihood gap must be >=0.05 to commit to a label.
This resolves the planning ambiguity about format selection without repeatedly
testing formats against the calibration gate. The three formats and selection
rule are published before any new observer results.

If the gate fails, stop main preservation/attack/prose/intervention/semantic stages.
Do not declare weight information ineffective: the observer was not reliable
enough to identify the hypothesis. Power and timing sensitivity reports may still
be completed. Post-gate experiments require a new explicitly documented protocol.

## Preservation and held-out evaluation

All 2,304 edits: BF16, 32 new text prompts with full vocabulary logits at every
input position, and eight greedy generations up to 64 tokens. Identity repeat,
gate invariance, finite values, exact restoration and circuit serialization
checks are required. One variant per arm per held-out layer (96 total), selected
by hash before evaluation, also receives FP32 checks. Exact numerical equality
is required; failures remain in the attack denominator, with reason recorded.
This finite text suite does not prove all-input floating-point equivalence.

Observer tables are re-extracted **from the edited full checkpoint in memory**.
Store original/edited up/down rows as safetensors. Independent fresh observer
contexts get four withheld wording variants: 9,984 A/B scoring calls including
cached original responses. Use 96 circuits, four per layer selected by hash,
first candidate seed, original plus four edited arms and two wording variants
for 960 unconditioned JSON prose responses (claim, units, reason; 192-token cap).
Never inject the forced-choice label into a prose response. Invalid output is
not success and does not trigger retries with a changed prompt.

## Analysis and mechanism

Primary outcome per original/edited pair is
`original_correct AND edited_equals_target AND preservation_passed`.
Keep all planned cases in the denominator. Average repeated wording/seed outcomes
within circuits and aggregate by layer. Bootstrap the 24 layers 10,000 times.
W-S and W-M require both a >=15 percentage point estimate and a positive lower
bound of 97.5% marginal intervals (Bonferroni simultaneous coverage >=95%), plus
positive differences for the separately generated prose claims. W-R is secondary.
Record baseline accuracy, abstentions, likelihood changes, conditional success,
preservation failures, perturbation norms, runtime, tokens and GPU peak memory.
Power is a planning sensitivity simulation across discordance and within-layer
correlation assumptions; it does not certify power from the small calibration set.

For at most one successful W candidate per layer, choose the candidate successful
on most main wordings, then hash tie-break. On new wordings 4 and 5, test all 16
subsets of the four edited neurons, plus the matched R plan. Re-verify full-model
BF16 preservation for these interventions. Undoing a scale edit is **not** neuron
ablation. A single candidate requires correct original, wrong fully edited model,
wrong single edit, and corrected leave-one-out edit on both new wordings.
Otherwise report a group effect or failed replication. No claim that Gemma itself
contains a deceptive intention or a literally lying neuron.

Compare no defense, a joint-up/down instruction, canonical `d*u` coefficients,
and a four-output exact-evaluator tool result. Canonical prompts may be identical
before/after by construction; this demonstrates representation invariance, not
new observer reasoning. See [implementation invariance](https://proceedings.mlr.press/v70/sundararajan17a.html).

## Conditional semantic extension

Only if primary analysis succeeds, generate 96 seeded false-user-claim prompts
(addition, multiplication and number comparison), development 32 / test 64.
Randomize the agree/disagree A/B mapping by alternating construction labels before
shuffling. Truth is original Gemma's conditional A/B likelihood, not mathematical
correctness. Target near-ties are ineligible and remain non-success denominator
cases. Provide the observer the prompt and fixed partial static MLP evidence:
normalized response-token embedding directions A/B and normalized unembedding
A-minus-B readout, projected onto 16 real units, with input scale 16. For multi-token
labels, use the final token direction and record the tokenization; likelihood
scoring uses the full equal-length completion. Do not expose target output or
activations. This does not represent the entire 4B model.

Require >=80% original-behavior prediction accuracy on the 32 development cases.
Otherwise stop. Test first-seed W/S/M/R edits on 64 held-out prompts without
retuning; verify the actual semantic prompt logits as well as the common suite.
Report exploratory effects, not a second independently powered main experiment.

## Execution and time budget

Use the installed full runtime; never modify the existing observer project:

```bash
export WSP_PYTHON=/media/jeong/ExtHDD/Gemma4_neuron_LAB/.venv/bin/python
python3 scripts/run_stage.py v1-preregister -- "$WSP_PYTHON" scripts/run_study.py preregister
python3 scripts/run_stage.py v1-extract -- "$WSP_PYTHON" scripts/run_study.py extract
python3 scripts/run_stage.py v1-calibrate -- "$WSP_PYTHON" scripts/run_study.py calibrate
# Only following a passing gate:
python3 scripts/run_stage.py v1-prepare -- "$WSP_PYTHON" scripts/run_study.py prepare
python3 scripts/run_stage.py v1-bf16 -- "$WSP_PYTHON" scripts/run_study.py preserve-bf16
python3 scripts/run_stage.py v1-fp32 -- "$WSP_PYTHON" scripts/run_study.py preserve-fp32
python3 scripts/run_stage.py v1-main -- "$WSP_PYTHON" scripts/run_study.py main
python3 scripts/run_stage.py v1-prose -- "$WSP_PYTHON" scripts/run_study.py prose
python3 scripts/run_stage.py v1-analyze -- "$WSP_PYTHON" scripts/run_study.py analyze
python3 scripts/run_stage.py v1-interventions -- "$WSP_PYTHON" scripts/run_study.py interventions
python3 scripts/run_stage.py v1-semantic -- "$WSP_PYTHON" scripts/run_study.py semantic
```

Run `power` for sensitivity simulation and `report` for regeneration. `--limit N`
on preservation/main/prose processes at most N additional trials; rerun without
changing inputs to resume. Journals atomically rewrite chunks of at most 32 rows,
validate provenance and reject duplicate/conflicting IDs. Frozen scientific source
changes are refused; fixing a scientific implementation requires an explicit new
registration/version, not silently reusing results. GPU runs need device access.

| Stage | Planned hours |
|---|---:|
| Preregistration | 2–3 |
| Harness and tests | 6–10 |
| Calibration/timing | 2–4 |
| Policy freeze | 2–4 |
| Preservation | 4–8 |
| Main binary evaluation | 6–12 |
| Prose | 4–8 |
| Reversal/defenses | 3–5 |
| Semantic extension, conditional | 6–10 |
| Analysis/report | 3–5 |
| Total | 38–69; contingency 50–90 |

Measure score latency (median/p90), generation speed, cold loading and preservation
separately; do not multiply a generation-free score measurement into prose time.
The first 50 scoring calls are included in timing. Planned 20-edit/50-prose timing
probes are conditional on a valid observer: if calibration fails they are not
launched merely to fill the timing table. Prose and preservation remain estimated.

After **each stage**, publish logs, config, reports and five evidence panels through
the GitHub connector and verify remote hashes. No terminal credentials or daemon
are currently configured. The agent drains queued snapshots before the next stage.
Use unrun panels with explicit NOT RUN labels, never fabricated zero-valued results.

## Interpretation

- W exceeds controls under all criteria: evidence for aligned weight information.
- Explanations change but W does not exceed controls: representation sensitivity.
- Calibration fails: the observer is inadequate for this protocol.
- Behavior preservation fails: that trial is not a valid explanation-only attack.

All raw failures, invalid responses and stopping decisions remain public. Initial
pilot outputs in the repository predate this preregistration and remain separate.
