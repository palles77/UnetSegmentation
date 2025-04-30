import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

from math import floor
import random
from Common.FileTools import *
from Common.StdOutCapture import *
from Learning.BestModelCheckpoint import BestModelCheckpoint
from ImageOperations.MaskFinder import is_window_outside_mask
from Learning.TrainingAndValidationDataGenerator import TrainingAndValidationDataGenerator
from Learning.WindowConversion import one_hot_encoding_to_three_channels, three_channels_to_one_hot_encoding
from Parameters.DefaultParameters import *
from ImageOperations.Contrast import *

from datetime import datetime
from os import path
from random import randint
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor, to_pil_image

class Unet:
    """Class with a UNET CNN network"""

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def __init__(self,                  
                 cml_args):
        """
        Initialize MyClass with input arguments.

        Args:
            cml_args: Command line arguments
        """

        # set instance variables for image and window properties
        super(Unet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.width = cml_args.width        
        self.height = cml_args.height
        self.window_size = cml_args.window_size
        self.windows_per_image_on_average = cml_args.windows_per_image_on_average
        self.percentage_train = cml_args.percentage_train
        self.min_percentage_for_fei_window = cml_args.min_fei_prct_window
        self.max_percentage_for_non_fei_window = cml_args.max_non_fei_prct_window
        self.fei_window_percentage = cml_args.fei_window_percentage
        self.dropout = cml_args.dropout

        # set up UNET model
        self.model = self.setup_unet_model(self.window_size, cml_args.filters_count, cml_args.kernel_size)

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def get_crop_shape(self, target, refer):
        """
        Calculate cropping shape for merging layers in UNET.

        Args:
            target (tensorflow.Tensor): Target layer to merge.
            refer (tensorflow.Tensor): Reference layer to merge.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: Tuple of two tuples representing crop shapes for height and width.
        """
        # calculate crop shape for width
        cw = target.get_shape()[2] - refer.get_shape()[2]
        assert cw >= 0
        if cw % 2 != 0:
            cw1, cw2 = int(cw / 2), int(cw / 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)

        # calculate crop shape for height
        ch = target.get_shape()[1] - refer.get_shape()[1]
        assert ch >= 0
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2), (cw1, cw2)  
    
    #===============================================================#
    #                                                               #
    #===============================================================#


    #===============================================================#
    #                                                               #
    #===============================================================#   
    def setup_unet_model(self, window_size, filters_count=64, kernel_size=3):
        """
        Initialize a U-Net model architecture.

        Args:
            window_size (int): Size of a CNN window.
            filters_count (int, optional): Number of filters. Defaults to 64.
            kernel_size (int, optional): Convolution kernel size. Defaults to 3.

        Returns:
            tensorflow.keras.Model: U-Net model.
        """
    def setup_unet_model(self, window_size, filters_count=64, kernel_size=3):
        class InternalUnet(nn.Module):
            def __init__(self):
                super(InternalUnet, self).__init__()
                self.encoder1 = self.conv_block(3, filters_count, kernel_size)
                self.encoder2 = self.conv_block(filters_count, filters_count * 2, kernel_size)
                self.encoder3 = self.conv_block(filters_count * 2, filters_count * 4, kernel_size)
                self.encoder4 = self.conv_block(filters_count * 4, filters_count * 8, kernel_size)
                self.center = self.conv_block(filters_count * 8, filters_count * 16, kernel_size)
                self.decoder4 = self.conv_block(filters_count * 16 + filters_count * 8, filters_count * 8, kernel_size)
                self.decoder3 = self.conv_block(filters_count * 8 + filters_count * 4, filters_count * 4, kernel_size)
                self.decoder2 = self.conv_block(filters_count * 4 + filters_count * 2, filters_count * 2, kernel_size)
                self.decoder1 = self.conv_block(filters_count * 2 + filters_count, filters_count, kernel_size)
                self.final = nn.Conv2d(filters_count, 2, kernel_size=1)

            def forward(self, x):
                enc1 = self.encoder1(x)
                enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
                enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
                enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
                center = self.center(nn.MaxPool2d(2)(enc4))
                dec4 = self.decoder4(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(center), enc4], dim=1))
                dec3 = self.decoder3(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dec4), enc3], dim=1))
                dec2 = self.decoder2(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dec3), enc2], dim=1))
                dec1 = self.decoder1(torch.cat([nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(dec2), enc1], dim=1))
                return nn.Softmax(dim=1)(self.final(dec1))
            
            def conv_block(self, in_channels, out_channels, kernel_size, activation=nn.LeakyReLU, padding=None):
                if padding is None:
                    padding = kernel_size // 2  # Default padding to keep the same spatial dimensions
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                    activation(),
                    nn.BatchNorm2d(out_channels)
                )

        return InternalUnet()

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def how_many_fei_windows(self, data_y: np.ndarray, image_index: int, total_fei_pixels: int, total_fei_windows: int) -> float:
        """
        Returns how many FEI windows should be extracted from an image.

        Args:
            data_y (np.ndarray): Data with index = 0 as non-FEI class in last dimension, and with index = 1 as FEI class in last dimension.
            image_index (int): Image index.
            total_fei_pixels (int): How many total FEI pixels in all images.
            total_fei_windows (int): How many windows of FEI we take in total.

        Returns:
            float: The number of FEI windows that should be extracted from the image.
        """
        image_fei_pixels = (data_y[image_index, :, :, 1] > 0.0).sum()
        result = int(round((float(image_fei_pixels) / float(total_fei_pixels)) * float(total_fei_windows)))

        return result

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def setup_random_windows(self, 
                            windows_per_image_on_average: int, 
                            augmentations_count: int,
                            min_percentage_for_fei_window: float,
                            max_percentage_for_non_fei_window: float,
                            fei_window_percentage: int):        
        """
        Returns a tensor of all random training, segmented windows pairs
        
        Args:
        windows_per_image_on_average (int): How many random windows to pick from each image on average.
        min_percentage_for_fei_window (float): Minimum percentage for a window to cover part of a fei.
        max_percentage_for_non_fei_window (float): Maximum percentage for a window to be considered non fei.
        fei_window_percentage (int): Percentage [0:100] of fei windows per image.
        fei_images_dir (str): Directory to save the images.
        save_images (bool): Whether to save the images or not.
        
        self.x_data and self.y_data are modified
        """
        
        # Define the number of windows per image
        windows_count = self.images_count * windows_per_image_on_average
        
        # Create empty arrays to store the data for the windows
        self.data_x_windowed = np.zeros((windows_count * (augmentations_count + 1), self.window_size, self.window_size, 3), dtype=np.float32)
        self.data_y_windowed = np.zeros((windows_count * (augmentations_count + 1), self.window_size, self.window_size, 2), dtype=np.float32)

        # Calculate the total number of fei windows needed
        total_fei_windows = (windows_count * fei_window_percentage) / 100.0

        # Calculate the total number of fei pixels in the dataset
        total_fei_pixels = (self.data_y[:, :, :, 1] > 0.0).sum()

        # Initialise the window index to 0
        window_index = 0
        
        how_many_fei_windows_saved = 0
        
        # Loop through all the images in the dataset
        for image_index in range(0, self.images_count):

            print("Analysing training image {} out of {} images".format(image_index, self.images_count))

            # Count the number of fei and non-fei windows needed
            fei_windows = 0
            non_fei_windows = 0

            # Calculate the number of fei and non-fei windows needed for this image
            fei_windows_per_image = self.how_many_fei_windows(self.data_y, image_index, total_fei_pixels, total_fei_windows)
            how_many_fei_windows_saved += fei_windows_per_image
            non_fei_windows_per_image = windows_per_image_on_average - fei_windows_per_image

            # If there are more fei windows needed than available, set it to the maximum available
            if (non_fei_windows_per_image < 0):
                non_fei_windows_per_image = 0

            # Set the attempts count to 0
            windows_attempts = 0

            # Loop until we have the required number of windows or we have reached the maximum number of attempts
            while True:

                windows_attempts += 1

                # Calculate random window offsets
                x_offset = randint(0, self.width - self.window_size - 1)
                y_offset = randint(0, self.height - self.window_size - 1)

                # Extract the window from the image
                window_x = self.data_x[ image_index, 
                                        y_offset : y_offset + self.window_size,
                                        x_offset : x_offset + self.window_size, : ]   
                window_y = self.data_y[ image_index, 
                                        y_offset : y_offset + self.window_size,
                                        x_offset : x_offset + self.window_size, : ]
                
                window_outside_mask = is_window_outside_mask(window_x)
                if window_outside_mask:
                    continue

                is_window_a_fei = self.is_window_a_fei(window_y, min_percentage_for_fei_window)
                is_window_a_non_fei = self.is_window_a_non_fei(window_y, max_percentage_for_non_fei_window)

                if is_window_a_fei:
                    fei_windows += 1
                else:
                    if is_window_a_non_fei:
                        non_fei_windows += 1
                    else:
                        continue                
                
                if (fei_windows == fei_windows_per_image) and (non_fei_windows == non_fei_windows_per_image):
                    break
                
                if windows_attempts > 1000 * windows_per_image_on_average:
                    print(f"Could not find enough windows in image with index: {image_index}.")
                    print(f"fei windows: {fei_windows}, non fei windows: {non_fei_windows}")
                    break

                # we cannot allow to exceed the maximum number of any classes
                # windows
                if fei_windows > fei_windows_per_image:
                    # we do not allow further count on fei windows now
                    fei_windows = fei_windows_per_image
                    continue

                if non_fei_windows > non_fei_windows_per_image:
                    # we do not allow further count on fei windows now
                    non_fei_windows = non_fei_windows_per_image
                    continue

                if window_index < windows_count :
                    self.data_x_windowed[window_index * (augmentations_count + 1), :, :, :] = window_x
                    window_y_one_hot_encoding = three_channels_to_one_hot_encoding(window_y)
                    self.data_y_windowed[window_index * (augmentations_count + 1), :, :, :] = window_y_one_hot_encoding
                else :
                    break                   

                window_index += 1
                
        print(f"Total fei windows saved: {how_many_fei_windows_saved} out of {total_fei_windows}")

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def augment_data(self, 
                    fei_images_dir: str, 
                    min_fei_percentage: float, 
                    windows_count: int, 
                    augmentations_count: int, 
                    how_many_images_to_save: int = -1) -> None:
        """
        Augment the data using torchvision.transforms.

        Args:
            fei_images_dir: A string representing the directory path to save the augmented images.
            min_fei_percentage: A float representing the minimum percentage of foreground pixels (fei) required for image to considered fei.
            windows_count: An integer saying how many windows were generated.
            augmentations_count: An integer representing the number of augmentations to apply to each image.
            how_many_images_to_save: An optional integer representing the number of images to save. If -1, no images are saved.

        Returns:
            None
        """

        # Set the random seed
        seed = 42
        torch.manual_seed(seed)
        width = self.data_x_windowed.shape[2]        
        height = self.data_x_windowed.shape[1]
        
        # Define the transformations
        transform = T.Compose([
            T.RandomRotation(10),
            T.RandomResizedCrop((height, width), scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, shear=0.2),
            T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05)
        ])

        print("Augmenting data")
        
        window_y_three_channels = np.zeros((height, width, 3), dtype=np.float32)
        
        for i in range(0, windows_count):
            for j in range(0, augmentations_count):
                # Apply transform to x_data
                window_x = self.data_x_windowed[i]
                window_x_tensor = to_tensor(window_x)
                augmented_window_x = transform(window_x_tensor)
                self.data_x_windowed[i * (augmentations_count + 1) + j + 1] = to_pil_image(augmented_window_x).numpy()
                
                # Apply transform to y_data
                window_y_random = self.data_y_windowed[i]
                window_y_three_channels = one_hot_encoding_to_three_channels(window_y_random)
                window_y_tensor = to_tensor(window_y_three_channels)
                augmented_window_y = transform(window_y_tensor)
                self.data_y_windowed[i * (augmentations_count + 1) + j + 1] = three_channels_to_one_hot_encoding(to_pil_image(augmented_window_y).numpy())
        
            print(f"Augmentation progress: {i + 1}/{windows_count} images processed")
        
        # Pick how_many_images_to_save windows and save them as pictures.
        random_seed = datetime.now().microsecond
        random.seed(random_seed)
        
        fei_windows_count_expected = self.fei_window_percentage * how_many_images_to_save / 100
        non_fei_windows_count_expected = how_many_images_to_save - fei_windows_count_expected
        fei_windows_generated = 0
        non_fei_windows_generated = 0
        how_many_images_saved = 0
        
        # We might be interested in saving the images
        if how_many_images_to_save != -1:
            
            empty_or_create_directory(fei_images_dir)            
            while how_many_images_saved < how_many_images_to_save:
                
                random_index = random.randint(0, windows_count * (augmentations_count + 1) - 1)
                
                window_x = self.data_x_windowed[random_index]
                window_y_three_channels = one_hot_encoding_to_three_channels(self.data_y_windowed[random_index])
                
                is_window_fei = self.is_window_a_fei(window_y_three_channels, min_fei_percentage)
                prefix = 'fei' if is_window_fei else 'nonfei'
                
                if is_window_fei:
                    if fei_windows_generated < fei_windows_count_expected:
                        fei_windows_generated += 1
                    else:
                        continue
                else:
                    if non_fei_windows_generated < non_fei_windows_count_expected:
                        non_fei_windows_generated += 1
                    else:
                        continue
                
                image = Image.fromarray(np.uint8(window_x))
                image.save(path.join(fei_images_dir, f'{prefix}_{random_index:03d}_x.png'))
                
                image = Image.fromarray(np.uint8(window_y_three_channels))
                image.save(path.join(fei_images_dir, f'{prefix}_{random_index:03d}_y.png'))
                
                how_many_images_saved += 1

    #===============================================================#
    #                                                               #
    #===============================================================#  
    def weighted_categorical_crossentropy(self, class_weights):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

        def loss(y_true, y_pred):
            y_true_class = torch.argmax(y_true, dim=1)
            class_weights_device = class_weights.to(y_true_class.device)  # Move class_weights to the same device as y_true_class
            weights_per_pixel = class_weights_device[y_true_class]
            cce = nn.CrossEntropyLoss(reduction='none')(y_pred, y_true_class)
            weighted_cce = cce * weights_per_pixel
            return torch.mean(weighted_cce)

        return loss
                    
    #===============================================================#
    #                                                               #
    #===============================================================#   
    def train(self, 
            cml_args) -> str:
        """
        Trains a neural network using the given parameters and saves the resulting model in the given path.

        :param cml_args: The command line arguments.

        :return: The path to the resulting zip file containing the trained model and training logs.
        """
        
        epochs = cml_args.epochs
        learning_rate = cml_args.learning_rate
        zip_model_dir = path.join(cml_args.models_dir, f"unet_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        zip_model_path = path.join(zip_model_dir, cml_args.unet_model_file)
        redirected_output_path = f'{zip_model_path}.output.txt'
        empty_or_create_directory(zip_model_dir)

        training_generator = TrainingAndValidationDataGenerator(cml_args, is_training=True)
        validation_generator = TrainingAndValidationDataGenerator(cml_args, is_training=False)

        best_val_loss = float('inf')

        with open(redirected_output_path, 'w') as output_file:
            std_out_capture = StdOutCapture(output_file)
            sys.stdout = std_out_capture

            print("Starting training the model...")
            print('Modified parameters passed from command line are: ')
            for arg, value in vars(cml_args).items():
                print(f"  {arg} = {value}")

            class_weights = {
                0: np.float32(1.0 - self.fei_window_percentage / 100.0),
                1: np.float32(self.fei_window_percentage / 100.0)
            }
            class_weights = list(class_weights.values())

            self.model = self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = self.weighted_categorical_crossentropy(class_weights)

            for epoch in range(epochs):
                self.model.train()
                total_batches = len(training_generator)
                correct_train = 0
                total_train = 0
                for batch_idx, batch in enumerate(DataLoader(training_generator, batch_size=1, shuffle=True)):
                    inputs, labels = batch
                    inputs = inputs.squeeze(0)
                    labels = labels.squeeze(0)
                    inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(labels, outputs)
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(outputs.data, 1)
                    labels = torch.argmax(labels, dim=1)  # Convert labels to class indices
                    total_train += labels.nelement()
                    correct_train += (predicted == labels).sum().item()

                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{total_batches}, Loss: {loss.item():.5f}")

                train_accuracy = 100 * correct_train / total_train

                self.model.eval()
                correct_val = 0
                total_val = 0
                val_loss = 0
                with torch.no_grad():
                    for batch in DataLoader(validation_generator, batch_size=1, shuffle=False):
                        inputs, labels = batch
                        inputs = inputs.squeeze(0)
                        labels = labels.squeeze(0)
                        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
                        outputs = self.model(inputs)
                        val_loss += criterion(labels, outputs).item()

                        _, predicted = torch.max(outputs.data, 1)
                        labels = torch.argmax(labels, dim=1)  # Convert labels to class indices
                        total_val += labels.nelement()
                        correct_val += (predicted == labels).sum().item()

                val_accuracy = 100 * correct_val / total_val
                val_loss /= len(validation_generator)

                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.5f}, Validation Loss: {val_loss:.5f}, Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), zip_model_path)
                    print(f"Best model saved with validation loss: {val_loss:.5f}")

        zip_name = path.join(f'{zip_model_dir}.zip')
        items_to_zip = os.listdir(zip_model_dir)
        index = 0
        for item_to_zip in items_to_zip:
            item_to_zip_full_path = path.join(zip_model_dir, item_to_zip)
            items_to_zip[index] = item_to_zip_full_path
            index += 1
        create_zip(items_to_zip, zip_name)

        return zip_name    
    #===============================================================#
    #                                                               #
    #===============================================================#           
    def is_window_a_fei(self, window: np.ndarray, min_percentage_for_fei_window: float) -> bool:
        """
        Determines whether a given window contains at least the minimum percentage of pixels classified as FEI.

        :param window: A 3D window to check for FEI pixels.
        :param min_percentage_for_fei_window: The minimum percentage of pixels in the window that must be FEI.

        :return: True if the window contains at least the minimum percentage of FEI pixels, False otherwise.
        """

        window_pixels_count = np.shape(window)[0] * np.shape(window)[1]
        fei_pixels = (window[:, :, :] > 0.0).sum() / 3
        percentage = (fei_pixels * 100.0) / window_pixels_count
        if percentage >= min_percentage_for_fei_window:
            return True
        else:
            return False

    #===============================================================#
    #                                                               #
    #===============================================================# 
    def is_window_a_non_fei(self, window, max_percentage_for_non_fei_window: float) -> bool:
        """
        Check if a window contains no frontal eye images (FEI) based on the count of non-FEI pixels.

        Args:
        - window: a 3D array representing a window of an image, where each element represents a pixel.
        - param max_percentage_for_non_fei_window: The maximum percentage of pixels in the window that must be NON FEI.

        Returns:
        - a boolean value indicating whether the window contains no FEI or not.
        """

        window_pixels_count = np.shape(window)[0] * np.shape(window)[1]
        fei_pixels = (window[:, :, :] > 0.0).sum() / 3
        percentage = (fei_pixels * 100.0) / window_pixels_count
        if percentage <= max_percentage_for_non_fei_window:
            return True
        else:
            return False
            
    #===============================================================#
    #                                                               #
    #===============================================================#   
    def segment(self, 
                cml_args,
                input_image_name, 
                input_mask_image_name,
                output_image_grey_name, 
                output_image_binary_name, 
                step=5):
        """
        Segments an input image using offsets and generates two output images: a grayscale image and a binary image.

        Args:
            cml_args: Command line arguments.
            input_image_name (str): File path to the input image.
            input_mask_image_name (str): File path to the input mask image.
            output_image_grey_name (str): File path for the output grayscale image in PNG format.
            output_image_binary_name (str): File path for the output binary image in PNG format.
            segmented_partially_name_prefix (str): Prefix for the file names of the offset-based segmentation results.
            step (int, optional): Number of pixels in each step to take. Defaults to 5.


        Returns:
            None

        Raises:
            None
        """
        input_image = cv2.imread(input_image_name)
        input_image_width = input_image.shape[1]
        input_image_height = input_image.shape[0]
        input_image_array = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_mask_image = cv2.imread(input_mask_image_name)
        input_mask_image_array = np.asarray(input_mask_image) / 255.0
        padding = cml_args.padding_to_remove

        # the extended image array contains window_padding pixels from each corner
        input_image_extended_array = np.zeros((input_image_height + padding * 2, input_image_width + padding * 2, 3), dtype=np.float32)
        input_image_extended_array[padding:input_image_height + padding, padding:input_image_width + padding, :] = input_image_array / 255.0

        segmented_array_probabilities_sum = np.zeros((input_image_height, input_image_width), dtype=np.float32)
        segmented_array_probabilities_count = np.zeros((input_image_height, input_image_width), dtype=np.float32)
        segmented_array_probabilities_binary = np.zeros((input_image_height, input_image_width), dtype=np.float32)

        # fill the prob_rectangles_dict
        for offset_y in range(0, self.window_size, step):
            for offset_x in range(0, self.window_size, step):
                self.segment_with_offset(cml_args,
                                        input_image_height,
                                        input_image_width,
                                        offset_y,
                                        offset_x,
                                        padding,
                                        input_image_extended_array,
                                        segmented_array_probabilities_sum,
                                        segmented_array_probabilities_count)

        # binarisation
        binarisation_level: float = cml_args.threshold_level / 255.0
        segmented_array_probabilities_binary = segmented_array_probabilities_sum / segmented_array_probabilities_count
        for x in range(0, input_image_width):
            for y in range(0, input_image_height):
                if segmented_array_probabilities_binary[y, x] >= binarisation_level:
                    segmented_array_probabilities_binary[y, x] = 255.0
                else:
                    segmented_array_probabilities_binary[y, x] = 0

        # now convert the segmented_array into shades of grey image
        # now scale the 0 to max_fei_probability to 0 to 255
        segmented_array_probabilities_sum = segmented_array_probabilities_sum / segmented_array_probabilities_count * 255.0
        
        segmented_array_probabilities_sum *= input_mask_image_array[:, :, 0]
        segmented_array_probabilities_binary *= input_mask_image_array[:, :, 0]

        # extract only foreground (fei channel)     
        segmented_array_uint8 = np.uint8(segmented_array_probabilities_sum)
        cv2.imwrite(output_image_grey_name, segmented_array_uint8)
        segmented_array_probabilities_binary_uint8 = np.uint8(segmented_array_probabilities_binary)
        cv2.imwrite(output_image_binary_name, segmented_array_probabilities_binary_uint8)

    #===============================================================#
    #                                                               #
    #===============================================================#    
    def segment_with_offset(
        self, 
        cml_args,
        height,
        width, 
        offset_y,        
        offset_x,
        padding, 
        input_image_extended_array, 
        segmented_array_probabilities_sum, 
        segmented_array_probabilities_count):
        """
        Segments a part of the input image using the specified offset.

        Args:
            cml_args: The command line arguments.
            height (int): Height of the input image.            
            width (int): Width of the input image.            
            offset_x (int): Offset at which the segmentation window is moved to the right.
            offset_y (int): Offset at which the segmentation window is moved to the bottom.
            padding (int): Number of pixels to avoid inside the segmentation window.
            input_image_extended_array (numpy.ndarray): The extended image array that contains window_padding pixels from each corner.
            segmented_array_probabilities_sum (numpy.ndarray): 2-dimensional array [height, width] with sums of offsetted segmentations.
            segmented_array_probabilities_count (numpy.ndarray): 2-dimensional array [height, width] with counts of offsetted segmentations.

        Returns:
            None

        Raises:
            None
        """
        # Effective segmentation region in each window
        reduced_window_size = self.window_size - 2 * padding

        rows_count = int(floor((height - offset_y) / reduced_window_size))
        cols_count = int(floor((width  - offset_x) / reduced_window_size))

        windows_queue = []

        for y_row in range(rows_count):
            for x_col in range(cols_count):

                xmin = offset_x + x_col * reduced_window_size
                ymin = offset_y + y_row * reduced_window_size
                xmax = xmin + self.window_size
                ymax = ymin + self.window_size

                # Append (row_index, col_index, sub-image array)
                windows_queue.append((y_row, x_col, input_image_extended_array[ymin:ymax, xmin:xmax, :]))

                # Process batch if we hit batch_size or the last window
                if (len(windows_queue) == cml_args.batch_size) or \
                    (x_col == (cols_count - 1) and y_row == (rows_count - 1)):

                    # -----------------------------
                    # 1) Build a NumPy batch (channels-last)
                    # -----------------------------
                    # Segment the data by predicting the results for shifted set of windows.
                    # The prediction is done in batches.
                    windows = np.zeros((len(windows_queue), self.window_size, self.window_size, 3), dtype=np.float32)

                    for window_index, windows_queue_element in enumerate(windows_queue):
                        windows[window_index] = windows_queue_element[2]
                    # -----------------------------
                    # 2) Convert to channels-first for PyTorch
                    #    Shape becomes (batch_size, 3, window_size, window_size)
                    # -----------------------------
                    windows = np.transpose(windows, (0, 3, 1, 2))  # channels-last -> channels-first
                    windows_torch = torch.from_numpy(windows).float().to(self.device)

                    # -----------------------------
                    # 3) Inference with your PyTorch model
                    #    Output shape typically: (batch_size, num_channels, H, W)
                    # -----------------------------
                    # Ensure the model is on the same device
                    self.model.to(self.device)
                    
                    with torch.no_grad():
                        predictions_torch = self.model(windows_torch)

                    # Convert back to channels-last for your post-processing
                    # Now shape is (batch_size, window_size, window_size, num_channels)
                    predictions_torch = predictions_torch.permute(0, 2, 3, 1)
                    predictions = predictions_torch.cpu().numpy()

                    # -----------------------------
                    # Post-processing loop
                    # -----------------------------
                    for win_idx, (y_row_seg, x_col_seg, _) in enumerate(windows_queue):

                        # Recompute window corners
                        xmin = offset_x + x_col_seg * reduced_window_size
                        ymin = offset_y + y_row_seg * reduced_window_size
                        xmax = xmin + self.window_size
                        ymax = ymin + self.window_size

                        # Extract the relevant prediction sub-array
                        # Here we only use channel 1, and exclude the padding
                        predicted_window = predictions[win_idx]
                        predicted_fg = predicted_window[
                            padding : padding + reduced_window_size,
                            padding : padding + reduced_window_size,
                            1  # Example: "foreground" channel
                        ]

                        # Add these values into your accumulators
                        segmented_array_probabilities_sum[ymin:ymax, xmin:xmax] += predicted_fg
                        segmented_array_probabilities_count[ymin:ymax, xmin:xmax] += 1.0

                    # Clear the queue for the next batch
                    windows_queue.clear()

        # Avoid division by zero
        segmented_array_probabilities_count[segmented_array_probabilities_count == 0] = 1.0
