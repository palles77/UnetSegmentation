#!/usr/bin/env bash

# Define the Google Drive file ID, output path, and extraction path
gdrive_id="1evUdkjj0tH4Nl2mEvtNz4vtVROfI5CKb"
outputPath="downloaded_drhagis.zip"
extractPath="extracted_files"

# ---------------------------------------
# Function: download_file_from_gdrive
# ---------------------------------------
# Downloads a file from Google Drive using its file ID.
# ---------------------------------------
download_file_from_gdrive() {
  local file_id="$1"
  local outputPath="$2"

  echo "Downloading file from Google Drive (ID: $file_id)..."
  # This method handles the confirmation token for large files
  confirm=$(curl -sc /tmp/gcookie "https://drive.google.com/uc?export=download&id=${file_id}" | \
    grep -o 'confirm=[^&]*' | sed 's/confirm=//')
  curl -Lb /tmp/gcookie "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${file_id}" -o "${outputPath}"
  echo "File downloaded successfully to '${outputPath}'."
}

# ---------------------------------------
# Function: decompress_file
# ---------------------------------------
# Decompresses the zip file to the specified folder.
# ---------------------------------------
decompress_file() {
  local filePath="$1"
  local extractPath="$2"

  echo "Decompressing file '$filePath'..."
  unzip -o "$filePath" -d "$extractPath"
  echo "File decompressed successfully to '$extractPath'."
}

# ---------------------------------------
# Main Script Execution
# ---------------------------------------

# 1) Download the file from Google Drive if needed
if [[ -f "$outputPath" ]]; then
  echo "File '$outputPath' already exists. Skipping download."
else
  download_file_from_gdrive "$gdrive_id" "$outputPath"
fi

# 2) Create the target directory if it doesn't exist
if [[ ! -d "$extractPath" ]]; then
  mkdir -p "$extractPath"
fi

# 3) Decompress the zip file
decompress_file "$outputPath" "$extractPath"