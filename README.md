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

Refer to [data/README.md](data/README.md)

# Recipe

Note: Keep in mind that recipe below is a very quick training recipe. The recommended values are mentioned in each step, but one can use the default values stored in file [.vscode/launch.json](.vscode/launch.json) to test if the solution works.

* Step 1. Python: Clean up directories and files
Run this step to make sure all your data has been removed prior to your training.

* Step 2. Python: Generate tensor data for training step
In current version of [.vscode/launch.json](.vscode/launch.json) in line 19
```
                "--maximum-generation-img-count=50"
```
we decided to parse only 50 images out of 900 images available for training U-Net. This is to allow for quick testing of our solution. Under full conditions we are looking at values such as 100 or more (up to 900).
To run this step on a machine with GPU should take a few minutes.

* Step 3. Python: Train unet for vessels segmentation
In current version of [.vscode/launch.json](.vscode/launch.json) in line 33
```
                "--epochs=5"
```
we decided to use only 5 epochs for training U-Net. This is to allow for quick testing of our solution. Under full conditions we are looking at values such as 10, or more (typically 20). The result of this step is a model stored in a directory [models/unet_model_20250529_000149](models/unet_model_20250529_000149) - here is the name of the directory run at 2025/05/29 at 00:01:49. This value needs to be applied to line 55 of launch.json so we can proceed to step 4. 
```
                "--unet-model-file=data/models/unet_model_20250529_000149/unet_model_weights_vs.pt",
```
To run this step on a machine with GPU should take a few minutes.

* Step 4. Python: cnn vessels segmentation assessment data preparation

For step 4 we need to modify list of files which are considered for CNN training. We are normally training CNN under 300 images from FIVES data. Currently we have two files:
* [data/parsed_dataset/fives/cnn_train/list_all.dbl](data/parsed_dataset/fives/cnn_train/list_all.dbl) - 300 images for training CNN
* [data/parsed_dataset/fives/cnn_train/list_short.dbl](data/parsed_dataset/fives/cnn_train/list_short.dbl) - 25 images for training CNN
epochs. In our scenario we use list_short.dbl, hence the line 54 in launch.json looks like:
```
                "--segmentation-dbl-file=data/parsed_dataset/fives/cnn_train/list_short.dbl",
```
but typically we recommend to use list_all.dbl. 
To run this step on a machine with GPU might take around 20 minutes.

* Step 5. Python: cnn vessels segmentation training

For step 5 we need to modify line 70 in launch.json to have a value of 15. Typically this value can be even higher, for example 30.
```
                "--epochs=15",
```
The outcome of this step will be stored in a directory models. For example our CNN model was stored in [models/cnn_model_20250529_005118](models/cnn_model_20250529_005118). This step can take a few minutes on a machine with a GPU. 

* Step 6. Python: Segment vessels with trained UNET and trained CNN

In this step we will finally do the segmentation testing on a joint validation dataset (DRHAGIS, HRF and STARE - total of 105 images). For that purpose we need to apply model generated step 3 into launch.json into line 91
```
                "--unet-model-file=data/models/unet_model_20250212_002522/unet_model_weights_vs.pt",
```
Line 92 might need to be modified experimentally to work on GPUs with memory lower than 8GB. The value of batch_size set to 2048 can for example be changed to 1024 for a machine with GPU of 4 GB.
```
                "--batch-size=2048",
```
To speed up testing we modify typical values from lines 93 to 95 to the following
```
                "--segment-rough-percent-step-1=25",
                "--segment-rough-percent-step-2=5",
                "--segment-rough-interval=5"
```
Typical recommended values would be set to 20, 4 and 10 for these parameters.

* Step 7. Python: Segment complex vessels with trained UNET and trained CNN in batches of parameters
* Step 8. Python: Segment simple vessels with trained UNET and trained CNN in batches of parameters

## Last update
2025/05/22
