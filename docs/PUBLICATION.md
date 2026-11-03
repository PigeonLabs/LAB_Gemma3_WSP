# Stage publication

## Version 2 exception

For `v2-*` stages, `run_stage.py` invokes `report_v2.py` only. It never
regenerates v1 reports or `artifacts/source_manifest.json`, which remains frozen
v1 evidence. V2 scientific provenance is in its preregistration and per-stage
source hashes. V2 diagnosis is followed only by its CPU audit/report, not by
automatic development or calibration: the user declined recurring automation.
No heartbeat was created. The current stage's publication is handled in this
active conversation, not by a scheduled background job.

## During this Codex experiment

1. Run `python3 scripts/run_stage.py STAGE -- COMMAND ...`. The runner waits for
   the stage, captures stdout/stderr, records exit code and pre-run source hashes,
   regenerates RESULTS/README/SVG, and freezes an allowlisted snapshot under
   `.publication/`. Failed commands are recorded and queued as well.
2. The assistant immediately drains each pending snapshot through the authenticated
   GitHub connector. This is an agent-managed step, not a background daemon or an
   executable Python-to-connector bridge. No terminal Git credential is currently set.
3. Read the current `main` ref and tree at `PigeonLabs/LAB_Gemma3_WSP`. Preserve
   other remote files. Inspect collisions if another writer changed tracked files.
4. `publication.py export SNAPSHOT` supplies exact content from the frozen snapshot.
   Create blobs for binary files, and a tree from text content / binary blob SHAs.
   Reuse unchanged blobs by comparing their Git SHA-1s. Use the current remote tree
   as the base. Commit with the current head as parent; update the ref without force
   and with an expected-head check. On a lease conflict, reread and reconcile.
5. Fetch the resulting commit's complete recursive tree and the branch head.
   Verify every expected path/blob SHA. Save that tree locally, then run
   `python3 scripts/publication.py ack SNAPSHOT --commit SHA --tree TREE_JSON`.
   Only an acknowledged snapshot is published. The connector should verify that
   the supplied tree belongs to that commit before acknowledgement.

The initial empty repository needs one Contents API seed commit, followed by the
complete snapshot commit. Existing remote content is never removed automatically.
The seed is setup, not a claim that the experiment has been fully uploaded.

## Optional unattended Git transport

On a normal checkout with `origin` pointing to the named repository, branch `main`,
and working Git credentials, use:

```bash
python3 scripts/run_stage.py --publish git STAGE -- COMMAND ...
# Retry a frozen pending snapshot (requires working tree matching the snapshot)
python3 scripts/publication.py git /absolute/path/to/.publication/SNAPSHOT
```

No force push, token storage, external service, recurring schedule, or GitHub Actions
GPU runner is installed. A failed push leaves a snapshot and exits with nonzero
status; a successful experiment is not rerun just to retry publication. Files cannot
be changed between snapshot and Git publication. Existing staged changes are refused.

## Included and excluded

- Included: code, tests, plans, raw JSON responses/scores, stdout/stderr logs, model
  metadata and hashes, small edited-row/column safetensors, docs and SVG figures.
- Excluded: approximately 8.6 GB of base model weights, caches, environment files,
  private keys, and the local publication outbox/receipts.
- The explicit path/extension allowlist, size cap, symlink rejection and common
  credential-pattern scan run before snapshotting. They are safeguards rather than
  a proof that arbitrary future experiment output contains no private information.
- The per-run source hashes are captured before execution. `source_manifest.json`
  instead describes the latest repository sources; do not mistake it for historic
  source provenance. Initial experiments predate this automatic stage recorder.
- SIGINT/SIGTERM are recorded after the child exits. SIGKILL, power loss and disk
  failure cannot guarantee a final record or upload; inspect incomplete logs before
  starting another stage. Logs are flushed to disk, not streamed to GitHub token by token.

## Figure and report generation

`scripts/build_readme.py` reads the actual preflight and forced-choice JSON results.
The SVG uses counts/actual answers, preserves TIE, and labels the sample as one circuit.
`docs/README.template.md` owns the narrative; `docs/EXPERIMENT_PROTOCOL.md` preserves
the detailed initial protocol. Changes to templates and scripts are included in the
next snapshot. Existing raw experimental files must not be overwritten.
