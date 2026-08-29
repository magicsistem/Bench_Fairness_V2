#!/usr/bin/env bash
# Network-only bootstrap for the CEDIA login node; no scientific computation.
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
YOLO=$ROOT/vendor/yolov7
REVISION=a207844b1ce82d204ab36d87d496728d3d2348e7
mkdir -p "$ROOT/.cedia/wheelhouse" "$ROOT/models/yolov7"
compgen -G "$ROOT/.cedia/wheelhouse/*" >/dev/null
if [[ ! -d "$YOLO/.git" ]]; then git clone https://github.com/WongKinYiu/yolov7.git "$YOLO"; fi
git --git-dir="$YOLO/.git" --work-tree="$YOLO" fetch origin "$REVISION"
git --git-dir="$YOLO/.git" --work-tree="$YOLO" checkout --detach "$REVISION"
if [[ ! -f "$ROOT/models/yolov7/yolov7_training.pt" ]]; then
    curl -fL --retry 3 -o "$ROOT/models/yolov7/.yolov7_training.pt.tmp" \
        https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
    mv "$ROOT/models/yolov7/.yolov7_training.pt.tmp" "$ROOT/models/yolov7/yolov7_training.pt"
fi
echo V2_NETWORK_RESOURCES_FETCHED
