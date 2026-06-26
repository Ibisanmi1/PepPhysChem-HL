#!/usr/bin/env bash
# One-commit "orphan" snapshot for Hugging Face Spaces — no old Git history, so no binary blobs
# in the push pack (fixes: "rejected because it contains binary files" after git rm --cached).
#
# Prereq: working tree clean (commit or stash). Checkpoints/images may stay on disk if .gitignored.
#
# Usage (from repo root):
#   chmod +x scripts/hf_push_space_orphan.sh
#   ./scripts/hf_push_space_orphan.sh
#
# Your current branch (e.g. main) is NOT rewritten — only the Space remote main is replaced.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
  echo "Error: uncommitted changes. Commit or stash, then re-run."
  echo ""
  git status -sb
  exit 1
fi

CURRENT="$(git rev-parse --abbrev-ref HEAD)"
REMOTE_NAME="${HF_REMOTE_NAME:-hf}"
SPACE_GIT_URL="https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL"

if ! git remote get-url "${REMOTE_NAME}" &>/dev/null; then
  git remote add "${REMOTE_NAME}" "${SPACE_GIT_URL}"
  echo "Added remote '${REMOTE_NAME}'"
else
  git remote set-url "${REMOTE_NAME}" "${SPACE_GIT_URL}"
fi

TMP="__hf_space_orphan_$$"
git branch -D "${TMP}" 2>/dev/null || true

git checkout --orphan "${TMP}"
# Orphan still copies the index from HEAD — tracked .pt / output/ / images/ would stay
# staged and end up in the new commit. Drop the index, then re-stage only .gitignore-safe paths.
git rm -rf --cached . >/dev/null 2>&1 || true
git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit — nothing staged after git add -A."
  git checkout "${CURRENT}"
  git branch -D "${TMP}" 2>/dev/null || true
  exit 1
fi

if git diff --cached --name-only | grep -qE '\.(pt|pth|ckpt)$'; then
  echo "Error: weight/checkpoint files are still staged. They must be gitignored for HF Spaces."
  echo "Staged matches:"
  git diff --cached --name-only | grep -E '\.(pt|pth|ckpt)$' || true
  git checkout "${CURRENT}"
  git branch -D "${TMP}" 2>/dev/null || true
  exit 1
fi

git commit -m "Space: slim snapshot (single commit, no prior binary history)"

echo "Force-pushing to ${REMOTE_NAME}/main (replaces Space Git history) ..."
set +e
git push --force "${REMOTE_NAME}" "${TMP}:main"
push_status=$?
set -e

# Always return to the previous branch (push may fail on Hub YAML / hooks).
git checkout "${CURRENT}"
git branch -D "${TMP}" 2>/dev/null || true

if [[ "${push_status}" -ne 0 ]]; then
  echo ""
  echo "Push failed. If the remote mentioned YAML / short_description:"
  echo "  README frontmatter short_description must be ≤ 60 characters (see docs/hf-space/DEPLOY.md)."
  exit "${push_status}"
fi

echo ""
echo "Done. Space: https://huggingface.co/spaces/Ibisanmi1/PepPhysChem-HL"
echo "Local branch '${CURRENT}' unchanged. Add weights under Space → Files → checkpoints/ if needed."
