#!/usr/bin/env bash

autodl_die() {
  echo "ERROR: $*" >&2
  exit 2
}

autodl_normalize_variant() {
  case "${1,,}" in
    raw) echo raw ;;
    datafixed) echo datafixed ;;
    datafixed-withnegative|datafixed-agentretention)
      echo datafixed-withnegative
      ;;
    datafixed-withnegative-terminal|datafixed-agentretention-terminalfix|datafixed-terminal|terminalfix)
      echo datafixed-withnegative-terminal
      ;;
    *) autodl_die "unsupported variant '$1'; use raw, datafixed, datafixed-withnegative, or datafixed-withnegative-terminal" ;;
  esac
}

autodl_internal_c3_variant() {
  case "$(autodl_normalize_variant "$1")" in
    raw) echo raw ;;
    datafixed) echo datafixed ;;
    datafixed-withnegative) echo datafixed-agentretention ;;
    datafixed-withnegative-terminal) echo datafixed-agentretention-terminalfix ;;
  esac
}

autodl_normalize_algorithm() {
  case "${1,,}" in
    grpo|bnpo|dapo|drgrpo) echo "${1,,}" ;;
    dr_grpo|dr-grpo) echo drgrpo ;;
    *) autodl_die "unsupported algorithm '$1'; use grpo, bnpo, dapo, or drgrpo" ;;
  esac
}

autodl_require_bool() {
  case "$2" in
    true|false) ;;
    *) autodl_die "$1 must be true or false: $2" ;;
  esac
}

autodl_require_file() {
  [[ -s "$2" ]] || autodl_die "$1 is missing or empty: $2"
}

autodl_require_dir() {
  [[ -d "$2" ]] || autodl_die "$1 is not a directory: $2"
}

autodl_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}
