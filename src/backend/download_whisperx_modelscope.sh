#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-./models}"
WHISPERX_DIR="$MODEL_ROOT/whisperx"

python -m pip install -U modelscope

mkdir -p \
  "$WHISPERX_DIR/alignment" \
  "$WHISPERX_DIR/large-v3" \
  "$WHISPERX_DIR/medium" \
  "$WHISPERX_DIR/pyannote"

download_model() {
  local model_id="$1"
  local local_dir="$2"
  echo
  echo "==> Downloading: ${model_id}"
  echo "==> To: ${local_dir}"
  mkdir -p "$local_dir"
  modelscope download --model "$model_id" --local_dir "$local_dir"
}

# 1. Whisper ASR: large-v3
download_model \
  openai-mirror/whisper-large-v3 \
  "$WHISPERX_DIR/large-v3"

# 2. Whisper ASR: medium
download_model \
  openai-mirror/whisper-medium \
  "$WHISPERX_DIR/medium"

# 3. Alignment models
download_model \
  jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn \
  "$WHISPERX_DIR/alignment/wav2vec2-large-xlsr-53-chinese-zh-cn"

download_model \
  facebook/wav2vec2-base-960h \
  "$WHISPERX_DIR/alignment/wav2vec2-base-960h"

# 4. Pyannote diarization
if ! download_model pyannote/speaker-diarization-3.1 "$WHISPERX_DIR/pyannote/speaker-diarization-3.1"; then
  download_model mirror013/speaker-diarization-3.1 "$WHISPERX_DIR/pyannote/speaker-diarization-3.1"
fi

# 5. Pyannote speaker embedding
download_model \
  ModelBulider/wespeaker-voxceleb-resnet34-LM \
  "$WHISPERX_DIR/pyannote/wespeaker-voxceleb-resnet34-LM" || true

# 6. Pyannote segmentation
download_model \
  pyannote/segmentation-3.0 \
  "$WHISPERX_DIR/pyannote/segmentation-3.0" || true

echo
echo "Done. Current structure:"
find "$WHISPERX_DIR" -maxdepth 2 -type d | sort
