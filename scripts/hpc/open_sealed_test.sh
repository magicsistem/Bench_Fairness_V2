#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$ROOT"
tag=$(git describe --tags --exact-match HEAD)
[[ $tag == v2-scientific-freeze-* ]]
scripts/hpc/run_in_container.sh --cpu -- python scripts/freeze.py verify --repo . --freeze scientific_freeze.json
source_root=${SEALED_TEST_SOURCE:-$HOME/Thesis_Fitzpatrick/data/raw/isic2018_task1}
target=data/raw/isic2018_task1
test ! -e "$target/ISIC2018_Task1-2_Test_Input"
mkdir -p "$target"
rsync -a --ignore-existing "$source_root/ISIC2018_Task1-2_Test_Input/" "$target/ISIC2018_Task1-2_Test_Input/"
if [[ -d "$source_root/ISIC2018_Task1_Test_GroundTruth" ]]; then
    rsync -a --ignore-existing "$source_root/ISIC2018_Task1_Test_GroundTruth/" "$target/ISIC2018_Task1_Test_GroundTruth/"
else
    url=$(python -c 'import json; print(json.load(open("configs/methodology_v2.json"))["sealed_test_ground_truth_url"])')
    mkdir -p artifacts/test
    curl -fL --retry 3 -o artifacts/test/ISIC2018_Task1_Test_GroundTruth.zip "$url"
    unzip -n artifacts/test/ISIC2018_Task1_Test_GroundTruth.zip -d "$target"
fi
scripts/hpc/run_in_container.sh --cpu -- python scripts/prepare_data.py sealed-manifest \
  --images "$target/ISIC2018_Task1-2_Test_Input" --masks "$target/ISIC2018_Task1_Test_GroundTruth" \
  --freeze scientific_freeze.json --output artifacts/test/access_manifest.json
printf '{"schema_version":2,"tag":"%s","commit":"%s","job_id":"%s","opened_utc":"%s"}\n' \
  "$tag" "$(git rev-parse HEAD)" "${SLURM_JOB_ID:-none}" "$(date -u +%FT%TZ)" > artifacts/test/open_event.json
echo V2_SEALED_TEST_OPENED
