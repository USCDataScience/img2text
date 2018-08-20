#!/bin/bash
# Workspace
WORK_SPACE=${1}
MODEL_DIR="${1}/Model"

cd im2txt
# Run a TensorBoard server.
tensorboard --logdir="${MODEL_DIR}"