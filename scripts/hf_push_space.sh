#!/usr/bin/env bash
# Push current branch to your Hugging Face Space Git remote.
# Prereq: run scripts/hf_set_token.sh (or hf auth login) first.
#
# Usage (from repo root):
#   ./scripts/hf_push_space.sh              # push HEAD -> hf main
#   ./scripts/hf_push_space.sh my-branch    # push my-branch -> hf main
#
# If push is rejected (Space has commits you do not), either merge first (see docs/hf-space/DEPLOY.md)
# or overwrite the Space branch (only if you accept losing remote-only commits):
#   HF_PUSH_FORCE=1 ./scripts/hf_push_space.sh
#
# Space: https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SPACE_GIT_URL="https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL"
LOCAL_BRANCH="${1:-$(git rev-parse --abbrev-ref HEAD)}"
REMOTE_NAME="${HF_REMOTE_NAME:-hf}"

if ! git remote get-url "${REMOTE_NAME}" &>/dev/null; then
  git remote add "${REMOTE_NAME}" "${SPACE_GIT_URL}"
  echo "Added remote '${REMOTE_NAME}' -> ${SPACE_GIT_URL}"
else
  git remote set-url "${REMOTE_NAME}" "${SPACE_GIT_URL}"
  echo "Updated remote '${REMOTE_NAME}' -> ${SPACE_GIT_URL}"
fi

git fetch "${REMOTE_NAME}" 2>/dev/null || true

USE_FORCE=0
if [[ "${HF_PUSH_FORCE:-}" == "1" || "${HF_PUSH_FORCE:-}" == "yes" ]]; then
  USE_FORCE=1
  echo "HF_PUSH_FORCE set: pushing with --force-with-lease (remote-only commits may be dropped)."
fi

echo "Pushing ${LOCAL_BRANCH} -> ${REMOTE_NAME}/main ..."
_push() {
  if [[ "${USE_FORCE}" -eq 1 ]]; then
    git push --force-with-lease "${REMOTE_NAME}" "${LOCAL_BRANCH}:main"
  else
    git push "${REMOTE_NAME}" "${LOCAL_BRANCH}:main"
  fi
}
if ! _push; then
  echo ""
  echo "Push failed."
  echo ""
  echo "If the remote mentioned binary files / Xet / pre-receive hook:"
  echo "  HF scans the whole push (all reachable commits). If you already removed binaries from"
  echo "  the index but old commits still contain them, use one of:"
  echo "    ./scripts/hf_push_space_orphan.sh   # one slim commit to hf/main (recommended)"
  echo "    git filter-repo …                   # rewrite local history (see docs/hf-space/DEPLOY.md)"
  echo "  Also: .gitignore + git rm -r --cached … (docs/hf-space/DEPLOY.md)."
  echo ""
  echo "If the remote said fetch first / non-fast-forward (history mismatch):"
  echo "  Option A — merge Space into your repo, then push again:"
  echo "    git fetch ${REMOTE_NAME} && git merge ${REMOTE_NAME}/main --allow-unrelated-histories"
  echo "    # resolve conflicts, commit, then: ./scripts/hf_push_space.sh"
  echo "  Option B — replace Space main with your branch (destructive):"
  echo "    HF_PUSH_FORCE=1 ./scripts/hf_push_space.sh ${LOCAL_BRANCH}"
  exit 1
fi

echo "Done. Space will rebuild: https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL"
