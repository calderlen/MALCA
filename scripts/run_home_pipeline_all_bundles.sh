#!/usr/bin/env bash

set -u -o pipefail

BUNDLES_DIR="${1:-output_migrated_camera_field_20260606/bundles}"

if [ ! -d "$BUNDLES_DIR" ]; then
    echo "Error: bundles directory not found: $BUNDLES_DIR" >&2
    exit 1
fi

if ! command -v malca >/dev/null 2>&1; then
    echo "Error: 'malca' is not on PATH" >&2
    exit 1
fi

shopt -s nullglob
bundles=("$BUNDLES_DIR"/*.zip)
shopt -u nullglob

if [ ${#bundles[@]} -eq 0 ]; then
    echo "No bundle zip files found in $BUNDLES_DIR" >&2
    exit 1
fi

echo "Found ${#bundles[@]} bundle(s) in $BUNDLES_DIR"
echo "Running MALCA home pipeline sequentially"
echo "========================================"

success_count=0
failed_count=0

for bundle in "${bundles[@]}"; do
    bundle_name="$(basename "$bundle")"
    echo
    echo "[$(date '+%F %T')] Starting $bundle_name"
    echo "Command: malca pipeline --stage home --import-bundle \"$bundle\" -o -v"

    if malca pipeline --stage home --import-bundle "$bundle" -o -v; then
        success_count=$((success_count + 1))
        echo "[$(date '+%F %T')] Finished $bundle_name"
    else
        status=$?
        failed_count=$((failed_count + 1))
        echo "[$(date '+%F %T')] Failed $bundle_name (exit $status)" >&2
    fi
done

echo
echo "========================================"
echo "Completed. Success: $success_count  Failed: $failed_count"

if [ "$failed_count" -ne 0 ]; then
    exit 1
fi
