#!/usr/bin/env bash
# Crash-recovery wrapper around cloud/vast/run_matrix.sh: launches one fold at a time,
# skips folds that already passed the infrastructure gate, and retries a fold a bounded
# number of times only on a training-process crash (never on a real gate failure).
#
# This does not exist as a separate concept in run_matrix.sh: that script intentionally
# stops the matrix and preserves evidence on any failure (see its own usage text). This
# wrapper adds unattended restart on top of that, for the class of failure that is not a
# science/protocol decision -- a killed process, an OOM, a transient CUDA/driver error --
# while still stopping hard and refusing to guess past a real infrastructure or ADR
# 0005/0006 fold-0 detection-safety-gate failure.
#
# Re-running this script (e.g. after an SSH drop or instance reboot) is safe: it always
# re-checks each fold against scripts/check_matrix_fold.py before doing anything, so an
# already-passed fold is skipped rather than re-launched.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: cloud/vast/run_matrix_autocontinue.sh --repo-dir DIR [options]

Options (mostly forwarded to run_matrix.sh):
  --repo-dir DIR       Checkout to run from (required)
  --fold-start N       First fold to launch, default 0
  --fold-end N         Last fold to launch (inclusive), default 4
  --arms LIST          Space-separated arms, default: baseline lff caf full
  --seed N             Default 2026
  --python PATH        Python executable, default python
  --m5-record PATH     Forwarded to run_matrix.sh; default evidence/m5-budget-rung.json
  --max-retries N      Retries per fold after a training-process crash, default 2
  --state-dir DIR      Where per-attempt logs and status.json are written;
                       default DIR/evidence/matrix-autocontinue relative to --repo-dir
  --dry-run            Forward --dry-run to run_matrix.sh for fold-start only, then exit

Requires det_data and det_models, exactly as run_matrix.sh does. Intended to be run under
nohup/setsid/tmux so it survives an SSH disconnect; it holds no lock of its own, so do not
run two instances of it against the same --repo-dir/--arms concurrently.
EOF
}

die() { echo "error: $*" >&2; exit 2; }
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

repo_dir=
fold_start=0
fold_end=4
arms=(baseline lff caf full)
seed=2026
python_bin=python
m5_record=
m5_record_set=false
max_retries=2
state_dir=
dry_run=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir) [[ $# -ge 2 ]] || die "--repo-dir requires a value"; repo_dir=$2; shift 2 ;;
        --fold-start) [[ $# -ge 2 ]] || die "--fold-start requires a value"; fold_start=$2; shift 2 ;;
        --fold-end) [[ $# -ge 2 ]] || die "--fold-end requires a value"; fold_end=$2; shift 2 ;;
        --arms)
            shift
            arms=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                arms+=("$1")
                shift
            done
            ((${#arms[@]} > 0)) || die "--arms requires at least one arm"
            ;;
        --seed) [[ $# -ge 2 ]] || die "--seed requires a value"; seed=$2; shift 2 ;;
        --python) [[ $# -ge 2 ]] || die "--python requires a value"; python_bin=$2; shift 2 ;;
        --m5-record) [[ $# -ge 2 ]] || die "--m5-record requires a value"; m5_record=$2; m5_record_set=true; shift 2 ;;
        --max-retries) [[ $# -ge 2 ]] || die "--max-retries requires a value"; max_retries=$2; shift 2 ;;
        --state-dir) [[ $# -ge 2 ]] || die "--state-dir requires a value"; state_dir=$2; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$repo_dir" ]] || die "--repo-dir is required"
[[ -f "$repo_dir/cloud/vast/run_matrix.sh" ]] || die "--repo-dir is not a GCALF-Net checkout: $repo_dir"
[[ "$fold_start" =~ ^[0-4]$ ]] || die "--fold-start must be 0 through 4"
[[ "$fold_end" =~ ^[0-4]$ ]] || die "--fold-end must be 0 through 4"
[[ "$fold_start" -le "$fold_end" ]] || die "--fold-start must be <= --fold-end"
[[ "$max_retries" =~ ^[0-9]+$ ]] || die "--max-retries must be a non-negative integer"
[[ -n "${det_data:-}" ]] || die "det_data must name the data root"
[[ -n "${det_models:-}" ]] || die "det_models must name the model root"

# Kept in sync with cloud/vast/run_matrix.sh's own `task=` line rather than duplicated,
# so retargeting one cannot silently desync from the other.
task=$(sed -n 's/^task=//p' "$repo_dir/cloud/vast/run_matrix.sh" | head -n1)
[[ -n "$task" ]] || die "could not read task= from $repo_dir/cloud/vast/run_matrix.sh"

if [[ "$m5_record_set" == false ]]; then
    m5_record="$repo_dir/evidence/m5-budget-rung.json"
fi

if [[ -z "$state_dir" ]]; then
    state_dir="$repo_dir/evidence/matrix-autocontinue"
fi
mkdir -p "$state_dir"
status_file="$state_dir/status.json"

run_matrix_args=(--repo-dir "$repo_dir" --arms "${arms[@]}" --seed "$seed" --python "$python_bin")
if [[ -n "$m5_record" ]]; then
    run_matrix_args+=(--m5-record "$m5_record")
fi

if [[ "$dry_run" == true ]]; then
    "$repo_dir/cloud/vast/run_matrix.sh" "${run_matrix_args[@]}" --fold-start "$fold_start" --fold-end "$fold_start" --dry-run
    exit 0
fi

write_status() {
    local fold=$1 state=$2 attempt=$3 detail=$4
    cat > "$status_file" <<JSON
{
  "task": "$task",
  "arms": $(printf '%s\n' "${arms[@]}" | python3 -c 'import json,sys; print(json.dumps([l.rstrip() for l in sys.stdin]))'),
  "fold": $fold,
  "state": "$state",
  "attempt": $attempt,
  "detail": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$detail"),
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}

fold_already_passed() {
    local fold=$1
    "$python_bin" "$repo_dir/scripts/check_matrix_fold.py" "$task" --fold "$fold" \
        --arms "${arms[@]}" --models-root "$det_models" \
        ${m5_record:+--m5-record "$m5_record"} >/dev/null 2>&1
}

archive_failed_fold() {
    local fold=$1 attempt=$2
    local ts
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    for arm in "${arms[@]}"; do
        local output_dir="$det_models/$task/RetinaUNetV001_D3V001_3d_${arm}/fold${fold}"
        if [[ -e "$output_dir" ]]; then
            local archive_dir="$det_models/$task/RetinaUNetV001_D3V001_3d_${arm}/fold${fold}.failed-attempt${attempt}-${ts}"
            log "archiving failed run aside (not deleting): $output_dir -> $archive_dir"
            mv "$output_dir" "$archive_dir"
        fi
    done
}

for fold in $(seq "$fold_start" "$fold_end"); do
    if fold_already_passed "$fold"; then
        log "fold $fold already passed the infrastructure gate; skipping"
        write_status "$fold" "skipped_already_passed" 0 "fold already had passing checkpoints/gate on disk"
        continue
    fi

    attempt=0
    fold_done=false
    while [[ "$attempt" -le "$max_retries" ]]; do
        attempt=$((attempt + 1))
        log "fold $fold: attempt $attempt/$((max_retries + 1))"
        write_status "$fold" "running" "$attempt" "launching run_matrix.sh"
        attempt_log="$state_dir/fold${fold}-attempt${attempt}.log"

        set +e
        "$repo_dir/cloud/vast/run_matrix.sh" "${run_matrix_args[@]}" \
            --fold-start "$fold" --fold-end "$fold" \
            > "$attempt_log" 2>&1
        exit_code=$?
        set -e

        if [[ "$exit_code" -eq 0 ]]; then
            log "fold $fold: passed on attempt $attempt"
            write_status "$fold" "passed" "$attempt" "run_matrix.sh exited 0"
            fold_done=true
            break
        fi

        if grep -q "FAILED the infrastructure sanity gate" "$attempt_log"; then
            log "fold $fold: real infrastructure/config gate failure (scripts/check_matrix_fold.py) -- not retrying"
            write_status "$fold" "failed_gate" "$attempt" "check_matrix_fold.py failed; evidence preserved, see $attempt_log"
            echo "Fold $fold failed the infrastructure sanity gate. This is a real config/correctness" >&2
            echo "problem, not a crash, and must not be auto-retried. See $attempt_log" >&2
            exit 1
        fi

        if grep -q "Fold-0 detection safety gate failed" "$attempt_log"; then
            log "fold $fold: ADR 0005/0006 fold-0 detection safety gate failed -- stopping per protocol"
            write_status "$fold" "failed_detection_gate" "$attempt" "check_fold0_detection_gate.py failed; report grade as pre-registered null, see $attempt_log"
            echo "Fold 0 failed the ADR 0005/0006 detection safety gate. Per protocol: stop, do not" >&2
            echo "launch folds 1-4, report grade as the pre-registered null. See $attempt_log" >&2
            exit 1
        fi

        if grep -q "one or more arms exited non-zero" "$attempt_log"; then
            log "fold $fold: attempt $attempt crashed (arm exited non-zero); treating as transient"
            write_status "$fold" "crashed" "$attempt" "one or more arms exited non-zero, see $attempt_log"
            if [[ "$attempt" -le "$max_retries" ]]; then
                archive_failed_fold "$fold" "$attempt"
                continue
            fi
            log "fold $fold: exhausted $max_retries retries after training-process crashes -- stopping"
            write_status "$fold" "failed_retries_exhausted" "$attempt" "see $attempt_log and archived fold${fold}.failed-attempt* dirs"
            exit 1
        fi

        log "fold $fold: attempt $attempt failed for an unrecognized reason (likely a usage/argument error) -- stopping"
        write_status "$fold" "failed_unrecognized" "$attempt" "see $attempt_log"
        exit 1
    done

    [[ "$fold_done" == true ]] || die "internal error: fold $fold loop exited without a recorded outcome"
done

log "folds ${fold_start}-${fold_end} complete for arms: ${arms[*]}"
write_status "$fold_end" "all_folds_complete" 0 "folds ${fold_start}-${fold_end} passed for arms: ${arms[*]}"
