#!/usr/bin/env bash

# Define the URL, output path, and extraction path
url="https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip"
outputPath="downloaded_hrf.zip"
extractPath="extracted_files/hrf"

# ---------------------------------------
# Function: download_file
# ---------------------------------------
# Checks if the specified file already exists.
# If not, downloads it from the provided URL.
# ---------------------------------------
download_file() {
  local url="$1"
  local outputPath="$2"

  echo "Checking if file '$outputPath' already exists..."
  if [[ -f "$outputPath" ]]; then
    echo "File '$outputPath' already exists. Skipping download."
  else
    echo "Downloading file from '$url'..."
    curl -L -o "$outputPath" "$url"
    echo "File downloaded successfully to '$outputPath'."
  fi
}

# ---------------------------------------
# Function: decompress_file
# ---------------------------------------
# Decompresses the given .zip file to the specified folder.
# ---------------------------------------
decompress_file() {
  local filePath="$1"
  local extractPath="$2"

  echo "Decompressing file '$filePath'..."
  # Using unzip for zip files
  unzip -o "$filePath" -d "$extractPath"
  echo "File decompressed successfully to '$extractPath'."
}

# ---------------------------------------
# Main Script Execution
# ---------------------------------------

# 1) Download the file if needed
download_file "$url" "$outputPath"

# 2) Create the target directory if it doesn't exist
if [[ ! -d "$extractPath" ]]; then
  mkdir -p "$extractPath"
fi

# 3) Decompress the zip file
decompress_file "$outputPath" "$extractPath"
