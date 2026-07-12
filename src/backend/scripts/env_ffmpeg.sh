#!/usr/bin/env bash

# Keep ffmpeg/ffprobe discovery stable for API and workers. Provide explicit
# EM2MEM_FFMPEG_BIN / EM2MEM_FFPROBE_BIN values in .env when system PATH is
# not enough.

_em2mem_prepend_path() {
  local dir="$1"
  if [[ -n "$dir" && -d "$dir" ]]; then
    case ":${PATH:-}:" in
      *":$dir:"*) ;;
      *) export PATH="$dir:${PATH:-}" ;;
    esac
  fi
}

_em2mem_prepend_ld_library_path() {
  local dir="$1"
  if [[ -n "$dir" && -d "$dir" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$dir:"*) ;;
      *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
}

_em2mem_prepend_ld_preload() {
  local lib="$1"
  if [[ -n "$lib" && -f "$lib" ]]; then
    case ":${LD_PRELOAD:-}:" in
      *":$lib:"*) ;;
      *) export LD_PRELOAD="$lib${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
  fi
}

_em2mem_resolve_tool_binary() {
  local tool_name="$1"
  local configured="${2:-}"
  local resolved=""

  if [[ -n "$configured" && "$configured" == /* ]]; then
    printf '%s\n' "$configured"
    return
  fi

  if [[ -n "$configured" ]]; then
    resolved="$(command -v "$configured" 2>/dev/null || true)"
    if [[ -n "$resolved" ]]; then
      printf '%s\n' "$resolved"
      return
    fi
  fi

  for dir in \
    "${CONDA_PREFIX:+$CONDA_PREFIX/bin}" \
    "${MAMBA_ROOT_PREFIX:+$MAMBA_ROOT_PREFIX/bin}" \
    "${HOME:+$HOME/miniconda3/bin}" \
    "${HOME:+$HOME/miniforge3/bin}" \
    "${HOME:+$HOME/mambaforge/bin}"
  do
    if [[ -n "$dir" && -x "$dir/$tool_name" ]]; then
      printf '%s\n' "$dir/$tool_name"
      return
    fi
  done

  resolved="$(command -v "$tool_name" 2>/dev/null || true)"
  if [[ -n "$resolved" ]]; then
    printf '%s\n' "$resolved"
    return
  fi

  printf '%s\n' "${configured:-$tool_name}"
}

_em2mem_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_em2mem_root_dir="$(cd "$_em2mem_script_dir/.." && pwd)"

if [[ -n "${FFMPEG_HOME:-}" && -d "$FFMPEG_HOME" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    _em2mem_prepend_path "$VIRTUAL_ENV/bin"
    _em2mem_prepend_path "$FFMPEG_HOME"
  else
    _em2mem_prepend_path "$FFMPEG_HOME"
  fi
fi

if [[ -z "${EM2MEM_FFMPEG_BIN:-}" && -n "${FFMPEG_HOME:-}" && -x "$FFMPEG_HOME/ffmpeg" ]]; then
  export EM2MEM_FFMPEG_BIN="$FFMPEG_HOME/ffmpeg"
else
  export EM2MEM_FFMPEG_BIN="$(_em2mem_resolve_tool_binary ffmpeg "${EM2MEM_FFMPEG_BIN:-}")"
fi

if [[ -z "${EM2MEM_FFPROBE_BIN:-}" && -n "${FFMPEG_HOME:-}" && -x "$FFMPEG_HOME/ffprobe" ]]; then
  export EM2MEM_FFPROBE_BIN="$FFMPEG_HOME/ffprobe"
else
  export EM2MEM_FFPROBE_BIN="$(_em2mem_resolve_tool_binary ffprobe "${EM2MEM_FFPROBE_BIN:-}")"
fi

if [[ -z "${EM2MEM_FFMPEG_LIB_DIR:-}" ]]; then
  if [[ -n "${FFMPEG_HOME:-}" && -d "$FFMPEG_HOME/../lib" ]]; then
    export EM2MEM_FFMPEG_LIB_DIR="$(cd "$FFMPEG_HOME/../lib" && pwd)"
  elif [[ -n "${EM2MEM_FFMPEG_BIN:-}" ]]; then
    _em2mem_ffmpeg_bin_dir="$(cd "$(dirname "$EM2MEM_FFMPEG_BIN")" 2>/dev/null && pwd || true)"
    if [[ -n "$_em2mem_ffmpeg_bin_dir" && -d "$_em2mem_ffmpeg_bin_dir/../lib" ]]; then
      export EM2MEM_FFMPEG_LIB_DIR="$(cd "$_em2mem_ffmpeg_bin_dir/../lib" && pwd)"
    fi
  fi
fi

_em2mem_prepend_ld_library_path "${EM2MEM_FFMPEG_LIB_DIR:-}"

if [[ "${EM2MEM_PRELOAD_FFMPEG_IMAGE_LIBS:-1}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  _em2mem_prepend_ld_preload "${EM2MEM_FFMPEG_LIB_DIR:-}/libtiff.so.6"
  _em2mem_prepend_ld_preload "${EM2MEM_FFMPEG_LIB_DIR:-}/libjpeg.so.8"
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  _em2mem_site_packages=""
  for _em2mem_candidate in "$VIRTUAL_ENV"/lib/python*/site-packages; do
    if [[ -d "$_em2mem_candidate" ]]; then
      _em2mem_site_packages="$_em2mem_candidate"
      break
    fi
  done
  if [[ -n "$_em2mem_site_packages" ]]; then
    _em2mem_prepend_ld_library_path "$_em2mem_site_packages/torch/lib"
    for _em2mem_nvidia_lib in "$_em2mem_site_packages"/nvidia/*/lib; do
      _em2mem_prepend_ld_library_path "$_em2mem_nvidia_lib"
    done
  fi
fi

if [[ -d "$_em2mem_root_dir/.venv_whisperx" && "${VIRTUAL_ENV:-}" == "$_em2mem_root_dir/.venv_whisperx" ]]; then
  _em2mem_prepend_path "$_em2mem_root_dir/.venv_whisperx/bin"
fi

echo "[em2mem_ffmpeg_env] FFMPEG_HOME=${FFMPEG_HOME:-}"
echo "[em2mem_ffmpeg_env] EM2MEM_FFMPEG_BIN=${EM2MEM_FFMPEG_BIN}"
echo "[em2mem_ffmpeg_env] EM2MEM_FFPROBE_BIN=${EM2MEM_FFPROBE_BIN}"
echo "[em2mem_ffmpeg_env] EM2MEM_FFMPEG_LIB_DIR=${EM2MEM_FFMPEG_LIB_DIR:-}"
