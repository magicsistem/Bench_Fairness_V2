#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
VENV=$ROOT/.cedia/venv
SITE_PACKAGES=$VENV/site-packages
YOLO=$ROOT/vendor/yolov7
REVISION=a207844b1ce82d204ab36d87d496728d3d2348e7

mkdir -p "$SITE_PACKAGES"
test ! -L "$SITE_PACKAGES"
find "$SITE_PACKAGES" -mindepth 1 -delete
python3 -m pip install --disable-pip-version-check --target "$SITE_PACKAGES" --upgrade --no-index --no-deps \
    --find-links "$ROOT/.cedia/wheelhouse" --requirement "$ROOT/configs/hpc/requirements-cuda.txt"

test -d "$YOLO/.git"
git --git-dir="$YOLO/.git" --work-tree="$YOLO" checkout --detach "$REVISION"
test "$(git --git-dir="$YOLO/.git" --work-tree="$YOLO" rev-parse HEAD)" = "$REVISION"

mkdir -p "$ROOT/models/yolov7"
test -f "$ROOT/models/yolov7/yolov7_training.pt"
sha256sum "$ROOT/models/yolov7/yolov7_training.pt" > "$ROOT/models/yolov7/yolov7_training.pt.sha256"
mkdir -p "$ROOT/artifacts/environment"
PYTHONPATH="$SITE_PACKAGES" python3 -m pip freeze > "$ROOT/artifacts/environment/pip-freeze.txt"
sha256sum "${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}" > "$ROOT/artifacts/environment/container.sha256"
PYTHONPATH="$SITE_PACKAGES" python3 -m pip check
PYTHONPATH="$SITE_PACKAGES" python3 -c 'import cv2,einops,numpy,scipy,segmentation_models_pytorch,skimage,timm,torch,transformers,yaml'
echo V2_BOOTSTRAP_COMPLETE
