#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
VENV=${VENV_PATH:-$ROOT/.cedia/venv}
YOLO=$ROOT/vendor/yolov7
REVISION=a207844b1ce82d204ab36d87d496728d3d2348e7

if [[ ! -x "$VENV/bin/python" ]]; then
    python3 -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/python" -m pip install --disable-pip-version-check \
    --constraint "$ROOT/configs/hpc/core-constraints.txt" \
    --requirement "$ROOT/configs/hpc/requirements-cuda.txt"

if [[ ! -d "$YOLO/.git" ]]; then
    git clone https://github.com/WongKinYiu/yolov7.git "$YOLO"
fi
git --git-dir="$YOLO/.git" --work-tree="$YOLO" fetch origin "$REVISION"
git --git-dir="$YOLO/.git" --work-tree="$YOLO" checkout --detach "$REVISION"
test "$(git --git-dir="$YOLO/.git" --work-tree="$YOLO" rev-parse HEAD)" = "$REVISION"

mkdir -p "$ROOT/models/yolov7"
if [[ ! -f "$ROOT/models/yolov7/yolov7_training.pt" ]]; then
    curl -fL --retry 3 -o "$ROOT/models/yolov7/.yolov7_training.pt.tmp" \
        https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
    mv "$ROOT/models/yolov7/.yolov7_training.pt.tmp" "$ROOT/models/yolov7/yolov7_training.pt"
fi
sha256sum "$ROOT/models/yolov7/yolov7_training.pt" > "$ROOT/models/yolov7/yolov7_training.pt.sha256"
mkdir -p "$ROOT/artifacts/environment"
"$VENV/bin/python" -m pip freeze > "$ROOT/artifacts/environment/pip-freeze.txt"
sha256sum "${SIF_PATH:-$HOME/pytorch_24.01-py3.sif}" > "$ROOT/artifacts/environment/container.sha256"
"$VENV/bin/python" -m pip check
echo V2_BOOTSTRAP_COMPLETE
