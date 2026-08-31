#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$ROOT"
CEDIA_LOGIN=${CEDIA_LOGIN:-miguel.benavides__yachaytech.edu.ec@hpc.cedia.edu.ec}
CEDIA_PORT=${CEDIA_PORT:-22}
REMOTE_ROOT=${REMOTE_ROOT:-/home/miguel.benavides__yachaytech.edu.ec/Bench_Fairness_V2}
LEDGER=.cedia/jobs.tsv
WHEEL_PYTHON=${WHEEL_PYTHON:-/home/miguel/miniforge3/envs/tesis-sam/bin/python}
mkdir -p .cedia
ssh_cmd=(ssh -p "$CEDIA_PORT" "$CEDIA_LOGIN")

sync_code() {
    local commit
    commit=$(git rev-parse HEAD)
    "${ssh_cmd[@]}" "cd \"$REMOTE_ROOT\" && git fetch origin '+refs/heads/main:refs/remotes/origin/main' && git checkout main && git merge --ff-only origin/main && test \"\$(git rev-parse HEAD)\" = '$commit'"
    mkdir -p .cedia/wheelhouse
    "$WHEEL_PYTHON" -m pip download --no-deps --dest .cedia/wheelhouse --requirement configs/hpc/requirements-cuda.txt
    rsync -a --ignore-existing -e "ssh -p $CEDIA_PORT" .cedia/wheelhouse/ "$CEDIA_LOGIN:$REMOTE_ROOT/.cedia/wheelhouse/"
    "${ssh_cmd[@]}" "cd \"$REMOTE_ROOT\" && scripts/hpc/fetch_dependencies.sh"
}

submit() {
    local phase=$1 stage=$2 dependency=${3:-} option=()
    [[ -z $dependency ]] || option=(--dependency="afterok:$dependency")
    local job
    job=$("${ssh_cmd[@]}" "cd \"$REMOTE_ROOT\" && mkdir -p logs && sbatch --parsable ${option[*]} --export=ALL,PROJECT_ROOT=\"$REMOTE_ROOT\" scripts/stages/$stage")
    job=${job%%;*}
    [[ $job =~ ^[0-9]+$ ]] || { echo "invalid sbatch id for $stage: $job" >&2; return 1; }
    printf '%s\t%s\t%s\n' "$phase" "$stage" "$job" >> "$LEDGER"; printf '%s' "$job"
}

wait_job() {
    local phase=$1 final_job=$2 state job
    while :; do
        while IFS=$'\t' read -r recorded_phase stage job; do
            [[ $recorded_phase == "$phase" ]] || continue
            state=$("${ssh_cmd[@]}" "sacct -j '$job' --format=State -n -X | head -1 | tr -d ' '" 2>/dev/null || true)
            case $state in
                FAILED*|CANCELLED*|TIMEOUT*|OUT_OF_MEMORY*|NODE_FAIL*)
                    echo "$stage job $job ended $state" >&2
                    "${ssh_cmd[@]}" "scancel $(awk -F '\t' -v p="$phase" '$1==p {print $3}' "$LEDGER" | tr '\n' ' ')" || true
                    return 1 ;;
            esac
        done < "$LEDGER"
        state=$("${ssh_cmd[@]}" "sacct -j '$final_job' --format=State -n -X | head -1 | tr -d ' '" 2>/dev/null || true)
        [[ $state != COMPLETED ]] || return 0
        sleep 30
    done
}

pre_freeze() {
    local start=${1:-0} dependency="" job stage index
    local stages=(00_preflight.slurm 01_bootstrap.slurm 02_prepare_development.slurm 03_train_yolov7_cv.slurm \
        04_yolov7_oof.slurm 05_calibrate_roi.slurm 06_train_yolov7_final.slurm 07_yolov7_validation.slurm \
        08_segment_validation.slurm 09_rank_top3.slurm 10_color_validation.slurm 23_create_freeze.slurm)
    ((start < ${#stages[@]})) || return 0
    for ((index=start; index<${#stages[@]}; index++)); do
        stage=${stages[index]}
        job=$(submit pre "$stage" "$dependency"); dependency=$job
    done
    wait_job pre "$dependency"
}

publish_freeze() {
    test ! -e scientific_freeze.json
    rsync -a -e "ssh -p $CEDIA_PORT" "$CEDIA_LOGIN:$REMOTE_ROOT/scientific_freeze.json" scientific_freeze.json
    git add scientific_freeze.json
    git commit -m "Publish V2 scientific freeze"
    git push origin main
    local stamp tag
    stamp=$(python -c 'import json; print(json.load(open("scientific_freeze.json"))["created_utc"].replace("-","").replace(":","").split(".")[0].replace("+0000","").replace("T","T")+"Z")')
    tag="v2-scientific-freeze-$stamp"
    git tag -a "$tag" -m "V2 scientific freeze $stamp"
    git push origin "$tag"
    "${ssh_cmd[@]}" "cd \"$REMOTE_ROOT\" && git pull --ff-only origin main && git fetch --tags origin && git checkout --detach '$tag'"
}

post_freeze() {
    local start=${1:-0} dependency="" job stage index
    local stages=(11_open_test.slurm 12_yolov7_test.slurm 13_segment_test.slurm 14_analyze_test.slurm \
        15_generate_mst.slurm 16_yolov7_mst.slurm 16_segment_mst.slurm 17_analyze_mst.slurm 18_prepare_mskcc.slurm \
        19_yolov7_mskcc.slurm 20_segment_mskcc.slurm 21_color_mskcc.slurm 22_analyze_mskcc.slurm 24_finalize.slurm)
    ((start < ${#stages[@]})) || return 0
    for ((index=start; index<${#stages[@]}; index++)); do
        stage=${stages[index]}
        job=$(submit post "$stage" "$dependency"); dependency=$job
    done
    wait_job post "$dependency"
}

first_unfinished() {
    local phase=$1 index=0 recorded_phase stage job state
    while IFS=$'\t' read -r recorded_phase stage job; do
        [[ $recorded_phase == "$phase" ]] || continue
        state=$("${ssh_cmd[@]}" "sacct -j '$job' --format=State -n -X | head -1 | tr -d ' '" 2>/dev/null || true)
        [[ $state == COMPLETED ]] || { echo "$index"; return; }
        ((index+=1))
    done < "$LEDGER"
    echo "$index"
}

prune_phase() {
    local phase=$1 keep=$2 temporary
    temporary=$(mktemp .cedia/jobs.XXXXXX)
    awk -F '\t' -v p="$phase" -v keep="$keep" 'BEGIN{n=0} $1!=p {print; next} n<keep {print} $1==p {n++}' "$LEDGER" > "$temporary"
    mv "$temporary" "$LEDGER"
}

status() {
    test -f "$LEDGER" || { echo "No V2 jobs recorded"; return 0; }
    while IFS=$'\t' read -r phase stage job; do
        printf '%s\t%s\t%s\t' "$phase" "$stage" "$job"
        "${ssh_cmd[@]}" "sacct -j '$job' --format=State,ExitCode,NodeList,Elapsed -n -X | head -1" || true
    done < "$LEDGER"
}

case ${1:-} in
    all)
        test -z "$(git status --porcelain --untracked-files=no)"
        sync_code; : > "$LEDGER"; pre_freeze; publish_freeze; post_freeze ;;
    status) status ;;
    resume)
        sync_code
        if [[ ! -f $LEDGER ]]; then "$0" all
        elif git describe --tags --exact-match HEAD 2>/dev/null | grep -q '^v2-scientific-freeze-'; then
            start=$(first_unfinished post); prune_phase post "$start"; post_freeze "$start"
        else
            start=$(first_unfinished pre); prune_phase pre "$start"; pre_freeze "$start"; publish_freeze; post_freeze
        fi ;;
    *) echo "Usage: ./run.sh {all|status|resume}" >&2; exit 2 ;;
esac
