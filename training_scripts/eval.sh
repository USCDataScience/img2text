#!/bin/bash
# Workspace
WORK_SPACE=${1}
TR_RECORD_DIR="${1}/TFRecords"
VOCAB_FILE="${WORK_SPACE}/TFRecords/word_counts.txt"

MODEL_DIR="${1}/Model"

cd im2txt
bazel build -c opt //im2txt/...
# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/im2txt/evaluate \
  --input_file_pattern="${TR_RECORD_DIR}/test-?????-of-00008" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --log_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/eval" \
  --vocab_file="${VOCAB_FILE}"
