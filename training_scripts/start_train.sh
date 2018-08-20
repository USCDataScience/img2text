#!/bin/bash
# Workspace
WORK_SPACE=${1}
TR_RECORD_DIR="${1}/TFRecords"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${1}/Inception_V3/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${1}/Model"

# Build the model.
cd im2txt
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${TR_RECORD_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=${2} \
  --number_of_steps=${3}