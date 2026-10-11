#!/usr/bin/env bash
# GeoPhys setup: fetch the benchmarks and the released artefacts.
#
#   ./setup.sh              everything
#   ./setup.sh members      just the precomputed member arrays (downloads the Drive folder, ~1.2 GB) -- enough to
#                           reproduce every number in the paper
#   ./setup.sh videos       the MAGI-1 candidate pool (~1 GB) -- needed only if you
#                           want to regenerate the feature stores yourself
#   ./setup.sh selections   the candidate each Physics-IQ configuration picks
#   ./setup.sh benchmarks   LikePhys and IntPhys2
#
# Everything released with the paper lives in one Drive folder:
#   https://drive.google.com/drive/folders/1l8gnrSPCD2wxke8SbDU_rnze8wy5kM8B
set -euo pipefail

DRIVE_FOLDER_ID="1l8gnrSPCD2wxke8SbDU_rnze8wy5kM8B"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHAT="${1:-all}"

have() { command -v "$1" >/dev/null 2>&1; }

need_gdown() {
  if ! have gdown; then
    echo "installing gdown (Drive downloader)"
    pip install --quiet gdown
  fi
}

fetch_drive() {
  need_gdown
  echo "fetching the release archive from Drive (about 1.2 GB)"
  local url="https://drive.google.com/drive/folders/${DRIVE_FOLDER_ID}"
  # resume partial downloads where this gdown version supports it
  local resume=""
  if gdown --help 2>/dev/null | grep -q -- "--continue"; then resume="--continue"; fi
  for attempt in 1 2 3; do
    if gdown --folder ${resume} "${url}" -O "${ROOT}/_drive"; then
      return 0
    fi
    echo "  download interrupted (attempt ${attempt}/3), retrying..."
    sleep 5
  done
  echo "Drive download failed; re-run ./setup.sh to try again" >&2
  return 1
}

unpack() {
  local name="$1" dest="$2"
  local tarball="${ROOT}/_drive/${name}"
  if [ ! -f "$tarball" ]; then
    echo "  ${name} not found in the Drive archive; skipping"
    return
  fi
  mkdir -p "$dest"
  echo "  unpacking ${name} -> ${dest}"
  tar -xf "$tarball" -C "$dest"
}

case "$WHAT" in
  all|members|videos|selections)
    fetch_drive
    ;;
esac

case "$WHAT" in
  all|members)
    # member arrays: what detect/verify read to reproduce the tables
    unpack "members.tar.gz" "${ROOT}/members"
    # ground-truth tables
    unpack "data.tar.gz" "${ROOT}/data"
    ;;
esac

case "$WHAT" in
  all|selections)
    # the candidate each Physics-IQ configuration picks, one folder per configuration
    unpack "selections.tar" "${ROOT}/selections"
    ;;
esac

case "$WHAT" in
  all|videos)
    # the 16-candidate MAGI-1 24B pool, for regenerating feature stores
    unpack "v2v_candidates.tar" "${ROOT}/videos"
    ;;
esac

case "$WHAT" in
  all|benchmarks)
    echo "LikePhys and IntPhys2 are distributed by their authors:"
    echo "  LikePhys  https://github.com/yuanjianhao508/likephys"
    echo "  IntPhys2  https://github.com/facebookresearch/IntPhys2"
    echo "Download them, then point GEOPHYS_STORES at wherever you extract features."
    ;;
esac

cat <<'EOF'

Done. To reproduce the tables:

    python -m geophys.detect --members members/
    python -m geophys.verify --members members/

To use GeoPhys on your own videos:

    python -m geophys.select    --videos my_candidates/ --out picks.csv
    python -m geophys.pairs     --videos my_pairs/      --out verdicts.csv
    python -m geophys.score     --videos model_a/ model_b/ --out scores.csv
    python -m geophys.calibrate --pairs my_labelled_pairs/ --out configs/mine.json

Feature stores are written under GEOPHYS_STORES (default ~/geophys_stores).
EOF
