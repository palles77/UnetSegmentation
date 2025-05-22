# Hybrid U-Net Segmentation
Hybrid U-Net segmentation of vessels in fundus eye images

# Cloning

You need GIT-LFS installed in order to clone this repository

# Preparing the environment

1. Install Anaconda with the latest version of Python (in our case it was Python 3.8.20)
2. Create Anaconda environment
```
conda create -n unet-segmentation-article python=3.11.5
conda activate unet-segmentation-article
```
3. Make sure your Python version is 3.11 or newer
```
python --version
```
4.1. Install relevant requirements from the current directory on Windows:
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r src/requirements.txt
```
4.2. Install relevant requirements from the current directory on Linux:
```
bash src/pytorch_linux.sh
pip install -r src/requirements.txt
```

# Downloading the data

Refer to [data/Readme.md](data/Readme.md)

# Recipe

File [.vscode/launch.json](.vscode/launch.json)
* Step 1. Python: Clean up directories and files
Run this step to make sure all your data has been removed prior to your training.
* Step 2. Python: Generate tensor data for training step

* Step 3. Python: Train unet for vessels segmentation
* Step 4. Python: cnn vessels segmentation assessment data preparation
* Step 5. Python: cnn vessels segmentation training
* Step 6. Python: Segment vessels with trained UNET and trained CNN
* Step 7. Python: Segment complex vessels with trained UNET and trained CNN in batches of parameters
* Step 8. Python: Segment simple vessels with trained UNET and trained CNN in batches of parameters

## Last update
2025/05/22
