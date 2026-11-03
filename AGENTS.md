# Experiment and publication workflow

- Target: Gemma 3 4B IT. Preserve original checkpoints and all raw results.
- Run every experiment stage via `python3 scripts/run_stage.py STAGE -- COMMAND ...`.
  This captures stdout/stderr, exit status, timestamps and code hashes, refreshes
  RESULTS/README/figures, and queues an immutable publication snapshot.
- After EACH stage, drain `.publication/` through the connected GitHub tools to
  `PigeonLabs/LAB_Gemma3_WSP`, branch `main`, before starting the next stage.
  Follow `docs/PUBLICATION.md`. Publish failures and interrupted runs too.
- Verify the remote tree against the queued Git blob hashes, then acknowledge
  the receipt locally. Pending is not published. Never force-update remote refs.
- When local Git credentials are configured, `--publish git` supports unattended
  commit/push. This machine currently uses the authenticated GitHub connector;
  the assistant performs that final transport step. Do not claim a background
  uploader exists or silently request/print authentication tokens.
- Only publish the explicit allowlist in `scripts/publication.py`. Keep base
  model weights, credentials, caches, and publication queues out of Git.
- Rebuild README from stored evidence; do not turn pilot sensitivity into claims
  of causal weight-analysis advantage, whole-model explanation manipulation, or
  liar-neuron discovery. Preserve denominators, ties and invalid responses.
