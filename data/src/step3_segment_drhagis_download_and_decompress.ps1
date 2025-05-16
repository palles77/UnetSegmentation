# Define the Google Drive file ID, output path, and extraction path
$googleDriveFileId = "1evUdkjj0tH4Nl2mEvtNz4vtVROfI5CKb"
$outputPath = "downloaded_drhagis.zip"
$extractPath = "extracted_files"

# Function to download a public Google Drive file
Function Download-GoogleDriveFile {
    param (
        [string]$FileId,
        [string]$OutputPath
    )

    $url = "https://drive.google.com/uc?export=download&id=$FileId"
    $session = New-Object Microsoft.PowerShell.Commands.WebRequestSession

    # Initial request to get the confirmation token (if needed)
    $response = Invoke-WebRequest -Uri $url -WebSession $session -UseBasicParsing

    if ($response.Content -match 'confirm=([0-9A-Za-z_]+)') {
        $confirm = $matches[1]
        $downloadUrl = "https://drive.google.com/uc?export=download&confirm=$confirm&id=$FileId"
        Invoke-WebRequest -Uri $downloadUrl -OutFile $OutputPath -WebSession $session -UseBasicParsing
    } else {
        # For small files, direct download
        Invoke-WebRequest -Uri $url -OutFile $OutputPath -WebSession $session -UseBasicParsing
    }
    Write-Host "File downloaded successfully to $OutputPath"
}

# Function to decompress the file
Function Decompress-File {
    param (
        [string]$FilePath,
        [string]$ExtractPath
    )

    Write-Host "Decompressing file $FilePath..."
    Expand-Archive -Path $FilePath -DestinationPath $ExtractPath -Force
    Write-Host "File decompressed successfully to $ExtractPath"
}

# Main script execution
$fileExists = Test-Path $outputPath
Write-Host "Checking if file $outputPath already exists... and the result is: $fileExists"
if ($fileExists) {
    Write-Host "File $outputPath already exists. Skipping download."
} else {
    Write-Host "Downloading file from Google Drive..."
    Download-GoogleDriveFile -FileId $googleDriveFileId -OutputPath $outputPath
}

# Create the target directory if it doesn't exist
if (!(Test-Path -Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath | Out-Null
}

Decompress-File -FilePath $outputPath -ExtractPath $extractPath