#!/bin/bash
WORK_SPACE="$(pwd)/WORK_SPACE"
FLICKR8K_DATASET_DIR="${WORK_SPACE}/Flickr8k/Flickr8k_Dataset"
FLICKR8K_TEXT_DIR="${WORK_SPACE}/Flickr8k/Flickr8k_text/Flickr8k.token.txt"
FLICKR8K_PROCESSED_TEXT_DIR="${WORK_SPACE}/Flickr8k/Flickr8k_index"
FLICKR8K_PROCESSED_TEXT_NAME="index.json"
TF_RECORDS_OUTPUT_DIR="${WORK_SPACE}/TFRecords"

python2 build_flicker8k_data.py \
  --flicker8k_text_dir="${FLICKR8K_TEXT_DIR}" \
  --train_image_dir="${FLICKR8K_DATASET_DIR}" \
  --train_captions_file_dir="${FLICKR8K_PROCESSED_TEXT_DIR}" \
  --train_captions_file="${FLICKR8K_PROCESSED_TEXT_NAME}" \
  --output_dir="${TF_RECORDS_OUTPUT_DIR}" \
  --word_counts_output_file="${TF_RECORDS_OUTPUT_DIR}/word_counts.txt" \
  --train_shards=256 \
  --val_shards=4 \
  --test_shards=8
