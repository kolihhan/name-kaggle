#!/usr/bin/env bash
set -euo pipefail

# Configure an already-downloaded GitHub Actions runner as a one-job GPU runner.
# Required:
#   RUNNER_TOKEN   short-lived registration token from repo Settings > Actions > Runners
# Optional:
#   REPO_URL       defaults to this repository
#   RUNNER_NAME    defaults to hostname-v3-gpu
#
# Run this script FROM the unpacked actions-runner directory. The runner is
# registered with --ephemeral, so GitHub automatically deregisters it after
# exactly one job. Never commit or print RUNNER_TOKEN.

: "${RUNNER_TOKEN:?Set RUNNER_TOKEN to a short-lived GitHub runner registration token}"
REPO_URL="${REPO_URL:-https://github.com/kolihhan/name-kaggle}"
RUNNER_NAME="${RUNNER_NAME:-$(hostname)-v3-gpu}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This frozen v3 runner path requires Linux" >&2
  exit 2
fi
if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "This frozen v3 runner path requires x86_64" >&2
  exit 2
fi
if [[ ! -x ./config.sh || ! -x ./run.sh ]]; then
  echo "Run this from an unpacked GitHub Actions runner directory containing config.sh and run.sh" >&2
  exit 2
fi

nvidia-smi
python3 - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
p = torch.cuda.get_device_properties(0)
gib = p.total_memory / (1024 ** 3)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("gpu", p.name)
print("vram_gib", f"{gib:.2f}")
if gib < 15.0:
    raise SystemExit(f"need >=15 GiB VRAM, found {gib:.2f}")
PY

# Default labels provide self-hosted/linux/x64. Add only the custom gpu label.
./config.sh \
  --unattended \
  --ephemeral \
  --url "$REPO_URL" \
  --token "$RUNNER_TOKEN" \
  --name "$RUNNER_NAME" \
  --labels gpu \
  --work _work

unset RUNNER_TOKEN

echo "Ephemeral GPU runner registered as: $RUNNER_NAME"
echo "It will accept one matching job and then deregister automatically."
exec ./run.sh
