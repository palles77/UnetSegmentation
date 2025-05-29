#!/usr/bin/env bash

models=(
  "cnn_model_20250214_222252"
  "unet_model_20250212_002522",
  "cnn_model_20250214_222252.zip",
  "unet_model_20250224_183427.zip"
  "cnn_model_20250529_005118.zip",
  "unet_model_20250529_000149.zip"
)

for model in "${models[@]}"; do
  # Unzip into current directory
  unzip "$model.zip" -d . 

  # Move the extracted folder out of data/models/
  mv "data/models/$model" .
done

# Remove the data folder (and subfolders) after processing
rm -rf data
