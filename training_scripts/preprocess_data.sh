#!/bin/bash
#WORK_SPACE=$1
IMAGE_DATASET_DIR="${1}"
IMAGE_TEXT_DIR="${2}"
IMAGE_PREPROCESSED_TEXT_DIR="${3}"
IMAGE_PREPROCESSED_TEXT_OUTPUT="index.json"
#TF_RECORDS_OUTPUT_DIR="${WORK_SPACE}/TFRecords"
TF_RECORDS_OUTPUT_DIR="TFRecords"

python build_image_data.py \
  --flicker8k_text_dir="${IMAGE_TEXT_DIR}" \
  --train_image_dir="${IMAGE_DATASET_DIR}" \
  --train_captions_file_dir="${IMAGE_PREPROCESSED_TEXT_DIR}" \
  --train_captions_file="${IMAGE_PREPROCESSED_TEXT_OUTPUT}" \
  --output_dir="${TF_RECORDS_OUTPUT_DIR}" \
  --word_counts_output_file="${TF_RECORDS_OUTPUT_DIR}/word_counts.txt" \
  --train_shards=256 \
  --val_shards=4 \
  --test_shards=8