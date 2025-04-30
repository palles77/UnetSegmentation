import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

import os
import re
import glob
import datetime
import sys

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from Common.StdOutCapture import *
from Common.FileTools import create_zip

class CnnGenerator(Dataset):
    """
    A custom PyTorch Dataset that:
      - Loads black-and-white images from disk
      - Computes their Jaccard distance vs. a 'truth' image
      - Returns (image_tensor, jaccard_distance)
    """
    def __init__(self, image_size, data_pairs):
        """
        :param image_size: Tuple (width, height) to which images will be resized
        :param data_pairs: List of tuples (input_image_path, jaccard_distance)
        """
        self.image_size = image_size
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, label = self.data_pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # label is the Jaccard distance in [0,1]
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return image, label_tensor

class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network that
    predicts a single score in [0,1].
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Example architecture: 5 convolution blocks + fully-connected
        # Adjust as needed
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(256)

        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with a probability of 0.5
        
        # After 5 pool layers, if input is 512x512 -> output is 16x16
        # 256 channels * 16 * 16 = 65536
        # Adjust if using different input sizes
        self.fc1   = nn.Linear(256 * 16 * 16, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool(self.bn3(torch.relu(self.conv3(x))))
        x = self.pool(self.bn4(torch.relu(self.conv4(x))))
        x = self.pool(self.bn5(torch.relu(self.conv5(x))))

        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(torch.relu(self.fc1(x)))  # Apply dropout after first fully connected layer
        x = self.dropout(torch.relu(self.fc2(x)))  # Apply dropout after second fully connected layer
        x = self.fc3(x)
        x = torch.nn.functional.softplus(x)  # Use Softplus activation function
        x = x / (1 + x)  # Normalize to range [0, 1]
        return x
    
class Cnn:
    """
    Main class that:
      - Collects image paths
      - Calculates Jaccard distance
      - Builds a dataset
      - Trains a CNN to predict image quality in [0,1]
    """
    def __init__(self, images_dir, model_dir, model_file_name, epochs, batch_size, learning_rate):
        """
        :param images_dir: Directory with images for CNN model training
        :param model_dir:  Directory where a time-stamped folder will be created
                           to store the trained model
        """
        self.images_dir = images_dir
        self.model_dir = model_dir
        self.model_file_name = model_file_name

        # Prepare a unique folder for saving the model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(self.model_dir, "cnn_model_" + timestamp)
        self.redirected_output_path = os.path.join(self.save_dir, model_file_name + ".txt")

        # We'll store (image_path, jaccard_distance) pairs
        self.data_pairs = []
        
        # You can adjust these as needed
        self.image_size = (512, 512)
        self.num_workers = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
       
        # Build or define the model
        self.model = SimpleCNN()

    @staticmethod
    def compute_jaccard_distance(image_size, pred_img_path, truth_img_path):
        """
        Compute Jaccard distance between two black and white images:
          Jaccard Index  = intersection / union
          Jaccard Distance = 1 - Jaccard Index

        :param image_size: Size to which images will be resized
        :param pred_img_path: Path to predicted/modified image
        :param truth_img_path: Path to ground truth image
        :return: Float in [0,1] indicating Jaccard distance
        """        
        pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
        truth_img = cv2.imread(truth_img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale

        # scale pred_img the self.image_size
        pred_img = cv2.resize(pred_img, image_size, interpolation=cv2.INTER_LANCZOS4)
        truth_img = cv2.resize(truth_img, image_size, interpolation=cv2.INTER_LANCZOS4)

        # Convert to numpy arrays
        pred_arr = np.array(pred_img)
        truth_arr = np.array(truth_img)

        # Binarize: assume white is "object" => any pixel > 127 is 1, else 0
        pred_bin = (pred_arr > 127).astype(np.uint8)
        truth_bin = (truth_arr > 127).astype(np.uint8)

        intersection = np.logical_and(pred_bin, truth_bin).sum()
        union = np.logical_or(pred_bin, truth_bin).sum()

        if union == 0:
            # Edge case: if both are entirely black => jaccard index = 1 => distance = 0
            return 0.0

        jaccard_index = intersection / union
        return jaccard_index

    def prepare_data(self):
        """
        Enumerate files in images_dir, parse them, find each 'prefix_truth.png',
        and compute the Jaccard distance for the images with 'prefix_XX.png' 
        with respect to their prefix's truth image.
        """
        # Use a dictionary to group files by prefix
        # { prefix_str: { 'truth': path_to_truth, 'variants': [paths to prefix_XX] } }
        pattern = re.compile(r"(.+)_(\d{1,3}|truth)\.png$")
        all_files = glob.glob(os.path.join(self.images_dir, "*.png"))

        prefixes = {}

        for fpath in all_files:
            filename = os.path.basename(fpath)
            match = pattern.match(filename)
            if not match:
                # Skip files that don't match the pattern
                continue

            prefix, suffix = match.groups()
            if prefix not in prefixes:
                prefixes[prefix] = {'truth': None, 'variants': []}

            # Check if it's truth or a numeric variant
            if suffix == 'truth':
                prefixes[prefix]['truth'] = fpath

            prefixes[prefix]['variants'].append(fpath)

        # Now compute Jaccard distances
        for prefix, info in prefixes.items():
            truth_path = info['truth']
            variants = info['variants']

            # If there's no truth file, skip
            if not truth_path or len(variants) == 0:
                continue

            for variant_path in variants:
                jd = self.compute_jaccard_distance(self.image_size, variant_path, truth_path)
                # We'll store (image_path, jaccard_distance)
                self.data_pairs.append((variant_path, jd))

    def train_model(self):
        """
        Build the Dataset, DataLoader, and run the training loop for the CNN.
        """
        # Split data into training and validation sets
        train_pairs, val_pairs = train_test_split(self.data_pairs, test_size=0.2, random_state=42)
        
        # Prepare datasets and dataloaders
        train_dataset = CnnGenerator(self.image_size, train_pairs)
        val_dataset = CnnGenerator(self.image_size, val_pairs)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        best_val_loss = float('inf')
        best_model_state = None
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.redirected_output_path), exist_ok=True)

        with open(self.redirected_output_path, 'w') as output_file:
            std_out_capture = StdOutCapture(output_file)
            sys.stdout = std_out_capture
            for epoch in range(self.epochs):
                self.model.train()
                running_loss = 0.0

                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = self.model(images)  # shape [batch_size, 1]
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)

                epoch_loss = running_loss / len(train_dataset)
                print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {epoch_loss:.6f}")

                # Validation
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)

                val_loss /= len(val_dataset)
                print(f"Epoch [{epoch+1}/{self.epochs}], Validation Loss: {val_loss:.6f}")
                
                # Remember the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict()
                    print(f"Best model updated with validation loss: {val_loss:.6f}")

            # Load the best model state
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                print("Loaded the best model state.")

    def save_model(self):
        """
        Save the trained model's state_dict to the designated self.save_dir.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_path = os.path.join(self.save_dir, self.model_file_name)
        torch.save(self.best_model_state, self.model_path)
        
        model_parent_dir = os.path.dirname(self.model_path)
        model_parent_dir_base_name = os.path.basename(model_parent_dir)
        zip_file_name = os.path.join(self.model_dir, model_parent_dir_base_name + ".zip")
        files_from_save_dir = glob.glob(os.path.join(self.save_dir, "*"))        
        create_zip(files_from_save_dir, zip_file_name)
        
        print(f"Model saved to: {zip_file_name}")
        
    def predict(self, image_path):
        """
        Predict the Jaccard distance for a given image.

        :param image_path: Path to the input image
        :return: Predicted Jaccard distance
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        # Load and preprocess the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dimensions

        with torch.no_grad():
            output = self.model(image)
            jaccard_distance = output.item()

        return jaccard_distance        

    def run(self):
        """
        Helper method to run the entire pipeline:
          1) Prepare data
          2) Train model
          3) Save model
        """
        print("[INFO] Preparing data...")
        self.prepare_data()
        print(f"[INFO] Found {len(self.data_pairs)} training samples.")

        print("[INFO] Starting training...")
        self.train_model()

        print("[INFO] Saving model...")
        self.save_model()
