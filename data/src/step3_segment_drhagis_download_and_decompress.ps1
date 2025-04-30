# Define the URL, output path, and extraction path
$url = "http://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip"
$outputPath = "downloaded_drhagis.zip"
$extractPath = "extracted_files"

# Function to download the file
Function Download-File {
    param (
        [string]$Url,
        [string]$OutputPath
    )

    $fileExists = Test-Path $OutputPath
    Write-Host "Checking if file $OutputPath already exists... and the result is: $fileExists"
    if ($fileExists) {
        Write-Host "File $OutputPath already exists. Skipping download."
    } else {
        Write-Host "Downloading file from $Url..."
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -UseBasicParsing
        Write-Host "File downloaded successfully to $OutputPath"
    }
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
Download-File -Url $url -OutputPath $outputPath

# Create the target directory if it doesn't exist
if (!(Test-Path -Path $extractPath)) {
    New-Item -ItemType Directory -Path $extractPath | Out-Null
}

Decompress-File -FilePath $outputPath -ExtractPath $extractPath