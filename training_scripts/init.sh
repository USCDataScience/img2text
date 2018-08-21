##!/usr/bin/env bash
IM2TXT_URL="https://github.com/tensorflow/models/trunk/research/im2txt"
INCEPTION_V3_URL="http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz"

WORK_SPACE="${1}"
IMAGE_PREPROCESSED_TEXT_DIR="${2}"
INCEPTION_V3_DIR="${WORK_SPACE}/Inception_V3"

svn export "${IM2TXT_URL}"

mkdir -p "${IMAGE_PREPROCESSED_TEXT_DIR}"
mkdir -p "${INCEPTION_V3_DIR}"

wget "${INCEPTION_V3_URL}" && tar -xzvf inception_v3_2016_08_28.tar.gz -C "${INCEPTION_V3_DIR}" && rm -rf inception_v3_2016_08_28.tar.gz

cd im2txt
bazel build -c opt //im2txt/...
