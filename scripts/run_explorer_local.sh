#!/usr/bin/env bash
#
# Launch the SAE Feature Explorer demo on localhost. Two modes:
#
#   1. Default — downloads every model in configs/models.yaml from the HF
#      dataset repo into ./local_data/, plus the thumbnails tarball into
#      ./local_images/, then launches bokeh serve. Idempotent.
#
#   2. --synthetic — generates a tiny synthetic registry under ./demo_data/
#      (no internet, no HF account) and launches against it.
#
# Env-var overrides:
#   PORT          (default 5006)
#   REGISTRY      (default configs/models.yaml)
#   LOCAL_DATA    (default ./local_data)
#   LOCAL_IMAGES  (default ./local_images)
#   HF_TOKEN      (read from ~/.hf_token if unset; ok empty for public repos)

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
PORT="${PORT:-5006}"

if [[ "${1:-}" == "--synthetic" ]]; then
    DEMO_DIR="${DEMO_DIR:-${REPO_ROOT}/demo_data}"
    if [[ ! -f "${DEMO_DIR}/models.yaml" ]]; then
        python "${HERE}/build_demo_data.py" "${DEMO_DIR}"
    fi
    exec python "${HERE}/bootstrap_demo.py" \
        --registry  "${DEMO_DIR}/models.yaml" \
        --target    "${DEMO_DIR}" \
        --image-dir "${DEMO_DIR}/images" \
        --skip-images \
        --launch --port "${PORT}"
fi

REGISTRY="${REGISTRY:-${REPO_ROOT}/configs/models.yaml}"
LOCAL_DATA="${LOCAL_DATA:-${REPO_ROOT}/local_data}"
LOCAL_IMAGES="${LOCAL_IMAGES:-${REPO_ROOT}/local_images}"

if [[ -z "${HF_TOKEN:-}" && -f "${HOME}/.hf_token" ]]; then
    HF_TOKEN="$(cat "${HOME}/.hf_token")"
fi
export HF_TOKEN="${HF_TOKEN:-}"

exec python "${HERE}/bootstrap_demo.py" \
    --registry  "${REGISTRY}" \
    --target    "${LOCAL_DATA}" \
    --image-dir "${LOCAL_IMAGES}" \
    --launch --port "${PORT}"
