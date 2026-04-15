#!/usr/bin/env bash
#
# Apply patches for a single phase.
# Usage:
#   apply-phase.sh 1              # Apply phase 1 patches
#   apply-phase.sh 2 --dry-run    # Dry-run phase 2
#   apply-phase.sh 2 --reverse    # Reverse phase 2
#   apply-phase.sh all            # Apply all phases in order
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS_DIR="$(dirname "$SCRIPT_DIR")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PHASE_NAME=""
DRY_RUN=0
REVERSE=0

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <phase-number|all> [--dry-run|--reverse]"
    echo "Phases: 1=sycl-sync, 2=sycl-kernel, 3=vulkan-intel"
    exit 1
fi

PHASE_ARG="$1"
shift

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --reverse) REVERSE=1 ;;
    esac
done

apply_phase() {
    local phase_num="$1"
    local phase_dir=""

    case "$phase_num" in
        1) phase_dir="phase1-sycl-sync"; PHASE_NAME="SYCL Sync (graph + async)" ;;
        2) phase_dir="phase2-sycl-kernel"; PHASE_NAME="SYCL Kernel (VER_GEN + DMMV tuning)" ;;
        3) phase_dir="phase3-vulkan-intel"; PHASE_NAME="Vulkan Intel (Arc 140T Xe2 override)" ;;
        4) phase_dir="phase4-host-copy"; PHASE_NAME="Host copy (remove PVC blanket workaround)" ;;
        *)
            echo -e "${RED}Unknown phase: $phase_num${NC}"
            echo "Valid phases: 1, 2, 3, 4"
            exit 1
            ;;
    esac

    local patch_dir="$SCRIPT_DIR/$phase_dir"
    local target_repo="$REPOS_DIR/llama.cpp"

    if [[ ! -d "$target_repo" ]]; then
        echo -e "${RED}Repo not found: $target_repo${NC}"
        exit 1
    fi

    if [[ ! -d "$patch_dir" ]]; then
        echo -e "${YELLOW}No patch directory: $patch_dir${NC}"
        return 0
    fi

    local patches=($(ls "$patch_dir"/*.patch 2>/dev/null | sort))
    if [[ ${#patches[@]} -eq 0 ]]; then
        echo -e "${YELLOW}No patches in $patch_dir${NC}"
        return 0
    fi

    local action="APPLYING"
    local git_flag=""
    if [[ $DRY_RUN -eq 1 ]]; then
        action="CHECKING"
        git_flag="--check"
    fi
    if [[ $REVERSE -eq 1 ]]; then
        action="REVERSING"
        git_flag="-R"
        if [[ $DRY_RUN -eq 1 ]]; then
            git_flag="-R --check"
        fi
    fi

    local applied=0 failed=0

    for patch_file in "${patches[@]}"; do
        local patch_name=$(basename "$patch_file")
        echo -n -e "  ${BLUE}${action}${NC} ${phase_dir}/${patch_name} ... "

        if (cd "$target_repo" && git apply $git_flag "$patch_file" 2>/dev/null); then
            if [[ $DRY_RUN -eq 1 ]]; then
                echo -e "${GREEN}[OK dry-run]${NC}"
            else
                echo -e "${GREEN}[OK]${NC}"
            fi
            ((applied++)) || true
        else
            if [[ $DRY_RUN -eq 1 ]]; then
                echo -e "${RED}[FAIL dry-run]${NC}"
            else
                echo -e "${RED}[FAIL]${NC}"
            fi
            ((failed++)) || true
        fi
    done

    echo -e "  ${GREEN}$applied ok${NC}, ${RED}$failed failed${NC}"
    return $failed
}

if [[ "$PHASE_ARG" == "all" ]]; then
    echo -e "${BLUE}=== All Phases ===${NC}"
    if [[ $DRY_RUN -eq 1 ]]; then echo -e "(dry-run)"; fi
    if [[ $REVERSE -eq 1 ]]; then echo -e "(reversing in reverse order: 3, 2, 1)"; fi
    echo ""
    total_fail=0

    if [[ $REVERSE -eq 1 ]]; then
        for p in 4 3 2 1; do
            echo -e "${BLUE}[Phase $p]${NC}"
            apply_phase $p || ((total_fail++)) || true
            echo ""
        done
    else
        for p in 1 2 3 4; do
            echo -e "${BLUE}[Phase $p]${NC}"
            apply_phase $p || ((total_fail++)) || true
            echo ""
        done
    fi

    if [[ $total_fail -gt 0 ]]; then
        echo -e "${RED}$total_fail phase(s) had failures.${NC}"
        exit 1
    fi
    echo -e "${GREEN}All phases complete.${NC}"
else
    echo -e "${BLUE}=== Phase $PHASE_ARG: $PHASE_NAME ===${NC}"
    if [[ $DRY_RUN -eq 1 ]]; then echo -e "(dry-run)"; fi
    if [[ $REVERSE -eq 1 ]]; then echo -e "(reversing)"; fi
    echo ""
    apply_phase "$PHASE_ARG"
fi
