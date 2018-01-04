#!/usr/bin/env bash
FLICKR8K_DATASET_URL="http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip"
FLICKR8K_TEXT_URL="http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip"
IM2TXT_URL="https://github.com/tensorflow/models/trunk/research/im2txt"
INCEPTION_V3_URL="http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"

WORK_SPACE="$(pwd)/WORK_SPACE"
FLICKR8K_DATASET_DIR="${WORK_SPACE}/Flickr8k/Flickr8k_Dataset"
FLICKR8K_TEXT_DIR="${WORK_SPACE}/Flickr8k/Flickr8k_text"
FLICKR8K_PROCESSED_TEXT_DIR="${WORK_SPACE}/Flickr8k/Flickr8k_index"
INCEPTION_V3_DIR="${WORK_SPACE}/Inception_V3"

svn export "${IM2TXT_URL}"

mkdir -p "${FLICKR8K_DATASET_DIR}"
mkdir -p "${FLICKR8K_TEXT_DIR}"
mkdir -p "${FLICKR8K_PROCESSED_TEXT_DIR}"
mkdir -p "${INCEPTION_V3_DIR}"

wget "${INCEPTION_V3_URL}" && tar -xzvf inception_v3_2016_08_28.tar.gz -C "${INCEPTION_V3_DIR}" && rm -rf inception_v3_2016_08_28.tar.gz
wget "${FLICKR8K_DATASET_URL}" && unzip Flickr8k_Dataset.zip -d "${FLICKR8K_DATASET_DIR}" && rm -rf Flickr8k_Dataset.zip
wget "${FLICKR8K_TEXT_URL}" && unzip Flickr8k_text.zip -d "${FLICKR8K_TEXT_DIR}" && rm -rf Flickr8k_text.zip

cd im2txt
bazel build -c opt //im2txt/...