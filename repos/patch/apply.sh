#!/usr/bin/env bash
#
# apply.sh — Apply all Intel GPU diagnosis patches
#
# Delegates to apply-phase.sh. See that script for options.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Parse args to detect --reverse and pass through
REVERSE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --reverse) REVERSE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --status)
            # Status mode — check each patch
            echo "Patch status:"
            for phase_dir in "$SCRIPT_DIR"/phase*/; do
                phase_name=$(basename "$phase_dir")
                echo ""
                echo "  [$phase_name]"
                for patch_file in "$phase_dir"*.patch; do
                    [[ -f "$patch_file" ]] || continue
                    patch_name=$(basename "$patch_file")
                    target_repo="$(dirname "$SCRIPT_DIR")/llama.cpp"
                    if (cd "$target_repo" && git apply --check "$patch_file" 2>/dev/null); then
                        echo -e "    ${YELLOW}NOT APPLIED${NC} $patch_name"
                    elif (cd "$target_repo" && git apply --check -R "$patch_file" 2>/dev/null); then
                        echo -e "    ${GREEN}APPLIED${NC}    $patch_name"
                    else
                        echo -e "    ${RED}CONFLICT${NC}   $patch_name"
                    fi
                done
            done
            exit 0
            ;;
    esac
done

exec "$SCRIPT_DIR/apply-phase.sh" all "$@"
