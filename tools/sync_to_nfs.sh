#!/bin/bash
# Push the source of this repository to the shareable copy, leaving its data alone.
#
# There are two copies of this code: the git working tree (source only) and the copy
# handed to colleagues, which additionally holds the datasets, the codes and the
# trained operators. Edit in the git tree, commit, then run this to update the other.
#
# Only files tracked by git are copied, so data/, cache/, checkpoints/, figures/ and
# results/ on the destination are never touched.
#
#   bash tools/sync_to_nfs.sh              # to ~/nfs/iae/code-github
#   bash tools/sync_to_nfs.sh <dest>       # somewhere else
#   DRY_RUN=1 bash tools/sync_to_nfs.sh    # show what would change, copy nothing
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$HOME/nfs/iae/code-github}"
DRY="${DRY_RUN:+--dry-run}"

if [ ! -d "$DEST" ]; then
  echo "destination does not exist: $DEST" >&2
  echo "create it first, or pass another path as the first argument" >&2
  exit 1
fi
if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  echo "$REPO is not a git working tree -- run this from the git copy, not the shared one" >&2
  exit 1
fi

echo "source      $REPO  ($(git -C "$REPO" rev-parse --short HEAD))"
echo "destination $DEST"
[ -n "$DRY" ] && echo "(dry run -- nothing is copied)"

if ! git -C "$REPO" diff-index --quiet HEAD --; then
  echo
  echo "note: the working tree has uncommitted changes; they will be copied too:"
  git -C "$REPO" status --short | sed 's/^/      /'
fi

echo
git -C "$REPO" ls-files -z |
  rsync -rlptv --files-from=- --from0 $DRY "$REPO/" "$DEST/"

# Files that used to be tracked and are not any more linger on the destination;
# rsync is deliberately not given --delete, so report them instead of removing them.
echo
extra=$(comm -13 \
  <(git -C "$REPO" ls-files | sort) \
  <(cd "$DEST" && find . -type f \
      -not -path './data/*' -not -path './cache/*' -not -path './checkpoints/*' \
      -not -path '*/figures/*' -not -path '*/results/*' -not -path '*/__pycache__/*' \
      -not -path './.git/*' | sed 's|^\./||' | sort))
if [ -n "$extra" ]; then
  echo "present on the destination but no longer tracked -- remove by hand if stale:"
  echo "$extra" | sed 's/^/      /'
else
  echo "no untracked leftovers on the destination."
fi
