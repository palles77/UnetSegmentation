import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

import string
import time
from Learning.Cnn import Cnn
from ImageOperations.ContrastDefault import process_image_with_default_contrast
from ImageOperations.Regions import count_regions_and_pixels
from Learning.PretrainingDataGenerator import *
from Learning.Unet import *
from Parameters.DefaultParameters import *
from Common.FileTools import *
from ImageOperations.SingleRegionFinder import get_brightest_region, save_brightest_region
from ImageOperations.MaskFinder import binarize_image
import os

#--------------------------------------------------------
# pre segment images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# padding_to_remove - how many pixels to avoid on border
#--------------------------------------------------------
def pre_segment_image_into_region(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    rough_sliding_step_1 = cml_args.segment_rough_sliding_step_1
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])
   
    #--------------------------------------------------------#
    # rough segmentation                                     #
    #--------------------------------------------------------#     
    percentage_step = cml_args.segment_rough_percent_step_1
    min_percent = cml_args.segment_min_percent_scale
    max_percent = cml_args.segment_max_percent_scale
    
    if (width / unet.window_size) < 6 or (height / unet.window_size) < 6:
        # If the image is too small, we need to increase the max_percent twice
        max_percent = (int)(max_percent * 1.5)
        rough_sliding_step_1 = (int)(rough_sliding_step_1 / 2)
        
    perform_rough_segmentation(unet, cml_args, temp_dir_name, input_full_file_name, 
                               rough_sliding_step_1, min_percent, max_percent, percentage_step)
    
    # We will take the segmented images and move them to the training directory
    # so we can calculate Jaccard distance between segmented binary files and target segmentation
    # to train the CNN model to understand which segmentation is better than others.
    # by associating scoring from 0 to 1 for each segmented image.
    segmented_pattern = os.path.join(temp_dir_name, 'output_*_segmented_binary*.png')
    segmented_binary_files = glob.glob(segmented_pattern)
    for segmented_binary_file_name in segmented_binary_files:
        input_short_file_name = os.path.basename(input_full_file_name)
        input_short_file_name_no_ext = os.path.splitext(input_short_file_name)[0]
        segmented_binary_file_name = os.path.basename(segmented_binary_file_name)
        percentage = int(segmented_binary_file_name.split('_')[1])
        destination_segmented_binary_file = input_short_file_name.replace(input_short_file_name_no_ext, f'{input_short_file_name_no_ext}_{percentage}') 
        destination_segmented_binary_file_full_path = os.path.join(cml_args.segmentation_training_dir, destination_segmented_binary_file)
        segmented_binary_file_name_full_path = os.path.join(temp_dir_name, segmented_binary_file_name)
        shutil.copy(segmented_binary_file_name_full_path, destination_segmented_binary_file_full_path)
        ground_truth_file = os.path.join(cml_args.segmentation_input_ground_dir, input_short_file_name)
        ground_truth_file_base_name = os.path.basename(ground_truth_file)
        destination_ground_truth_file = ground_truth_file_base_name.replace(".png", "_truth.png")
        destination_ground_truth_file_full_path = os.path.join(cml_args.segmentation_training_dir, destination_ground_truth_file)
        shutil.copy(ground_truth_file, destination_ground_truth_file_full_path)
        
#--------------------------------------------------------
# segment images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# padding_to_remove - how many pixels to avoid on border
#--------------------------------------------------------
def segment_image_into_region(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    consider_single_region = cml_args.consider_single_region
    consider_oval_region = cml_args.consider_oval_region
    eccentricity_level = cml_args.eccentricity_level
    rough_sliding_step_1 = cml_args.segment_rough_sliding_step_1
    rough_sliding_step_2 = cml_args.segment_rough_sliding_step_2
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])
   
    #--------------------------------------------------------#
    # rough segmentation                                     #
    #--------------------------------------------------------#     
    (segmented_grey_image_file_name, segmented_binary_image_file_name, _) = \
        get_segmented_and_grey_image_file_name(cml_args.segmentation_output_dir, input_short_file_name, file_name_suffix, rough_sliding_step_1)   

    percentage_step = cml_args.segment_rough_percent_step_1
    min_percent = cml_args.segment_min_percent_scale
    max_percent = cml_args.segment_max_percent_scale
    optional_cnn_model = None
    if os.path.exists(cml_args.cnn_model_file):
        optional_cnn_model = Cnn(cml_args.segmentation_training_dir, cml_args.models_dir, cml_args.cnn_model_file, 
                                 cml_args.epochs, cml_args.batch_size, cml_args.learning_rate)
        model_state_dict = torch.load(cml_args.cnn_model_file, weights_only=True)
        optional_cnn_model.model.load_state_dict(model_state_dict)
        optional_cnn_model.model.eval()
    
    if (width / unet.window_size) < 6 or (height / unet.window_size) < 6:
        # If the image is too small, we need to increase the max_percent twice
        max_percent = (int)(max_percent * 2.5)
        
    rough_percentage_stage_1 = perform_rough_segmentation(unet, cml_args, temp_dir_name, input_full_file_name,                                                           
                                                          rough_sliding_step_1, min_percent, max_percent, percentage_step, optional_cnn_model)

    #--------------------------------------------------------#
    # detailed segmentation                                  #
    #--------------------------------------------------------#
    percentage_step = cml_args.segment_rough_percent_step_2
    min_percent = rough_percentage_stage_1 - cml_args.segment_rough_interval
    max_percent = rough_percentage_stage_1 + cml_args.segment_rough_interval
    
    rough_percentage_stage_2 = perform_rough_segmentation(unet, cml_args, temp_dir_name, input_full_file_name, 
                                                          rough_sliding_step_2, min_percent, max_percent, percentage_step, optional_cnn_model)
    
    #--------------------------------------------------------#
    # final segmentation with very small step                #
    #--------------------------------------------------------#
    
    # If we have a rough percentage, we can do detailed segmentation.    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting final segmentation]"
    print(log_entry)
        
    (object_statistic, regions_count, detailed_grey_file_name, detailed_binary_file_name) = segment_image_by_percentage(
        unet, cml_args, temp_dir_name, input_full_file_name, 
        cml_args.segment_detailed_sliding_step, rough_percentage_stage_2, optional_cnn_model)
    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Finished final segmentation with statistic: {object_statistic}, with regions count: {regions_count}]"
    print(log_entry)

    scaled_down_grey_image = cv2.imread(detailed_grey_file_name)
    dim = (width, height)
    resized = cv2.resize(scaled_down_grey_image, dim, interpolation = cv2.INTER_LANCZOS4)
    # check if file exists, and delete it if it does
    if os.path.exists(segmented_grey_image_file_name):
        os.remove(segmented_grey_image_file_name)
    cv2.imwrite(segmented_grey_image_file_name, resized)

    scaled_down_binary_image = cv2.imread(detailed_binary_file_name)
    resized = cv2.resize(scaled_down_binary_image, dim, interpolation = cv2.INTER_LANCZOS4)
    if os.path.exists(segmented_binary_image_file_name):
        os.remove(segmented_binary_image_file_name)
    cv2.imwrite(segmented_binary_image_file_name, resized) 
    
    if consider_single_region:
        # We need to narrow down the results for the user
        brightest_region, _ = get_brightest_region(
            segmented_grey_image_file_name, segmented_binary_image_file_name, 
            eccentricity_level, consider_oval_region)
        
        if brightest_region is not None:
            
            log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Final segmentation image has been masked to the brightest region with bounding box: {brightest_region.bbox}]"
            print(log_entry)
            
            brightest_region_segmented_grey_image_file_name = save_brightest_region(brightest_region, segmented_grey_image_file_name)
            shutil.move(brightest_region_segmented_grey_image_file_name, segmented_grey_image_file_name)
            brightest_region_segmented_binary_image_file_name = save_brightest_region(brightest_region, segmented_binary_image_file_name)
            shutil.move(brightest_region_segmented_binary_image_file_name, segmented_binary_image_file_name)
        
    return (segmented_grey_image_file_name, segmented_binary_image_file_name)        

#--------------------------------------------------------
# segment simple images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# padding_to_remove - how many pixels to avoid on border
#--------------------------------------------------------
def segment_simple_image_into_region(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
   
    #--------------------------------------------------------#
    # final segmentation with very small step                #
    #--------------------------------------------------------#
    
    # If we have a rough percentage, we can do detailed segmentation.    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting final segmentation]"
    print(log_entry)
        
    (object_statistic, regions_count, detailed_grey_file_name, detailed_binary_file_name) = segment_image_by_percentage(
        unet, cml_args, temp_dir_name, input_full_file_name, 
        cml_args.segment_detailed_sliding_step, 100, None)
    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Finished final segmentation with statistic: {object_statistic}, with regions count: {regions_count}]"
    print(log_entry)
        
    return (detailed_grey_file_name, detailed_binary_file_name)

#--------------------------------------------------------
# segment images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# padding_to_remove - how many pixels to avoid on border
#--------------------------------------------------------
def segment_image_by_percentage(
                    unet,
                    cml_args,
                    temp_dir_name,
                    input_full_file_name,
                    sliding_window_step,
                    scale_percent,
                    optional_cnn_model = None):
    
    print(f"Scaling {input_full_file_name} to {scale_percent} percent.")
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])

    # Resize the image
    scaled_width = int(width * scale_percent / 100)
    scaled_height = int(height * scale_percent / 100)
    scaled_dim = (scaled_width, scaled_height)
    resized = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)       

    # Create a unique output file name based on the scale percent
    scaled_down_image_name = f"{temp_dir_name}/output_{scale_percent}.png"
    cv2.imwrite(scaled_down_image_name, resized)
    
    # Now let's have a masked version of the image    
    scaled_down_mask_image_name = f"{temp_dir_name}/output_{scale_percent}_bw.png"
    binarize_image(scaled_down_image_name, scaled_down_mask_image_name)
    
    scaled_down_normalized_image_name = f"{temp_dir_name}/output_{scale_percent}n.png"
    process_image_with_default_contrast(cml_args, scaled_down_image_name, scaled_down_normalized_image_name)    
    os.remove(scaled_down_image_name)
    shutil.move(scaled_down_normalized_image_name, scaled_down_image_name)   
    
    print(f"Segmentation start {scaled_down_image_name}")
    start_time = time.time()
    
    (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name) = \
        segment_image_no_contrast(unet, cml_args, scaled_down_image_name, scaled_down_mask_image_name, temp_dir_name, sliding_window_step)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Segmentation end {scaled_down_image_name}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    # Calculate how many white regions are in scaled_down_binary_file_name
    object_ratio = 0
    (regions_count, pixels_count) = count_regions_and_pixels(scaled_down_binary_segment_file_name)
    
    if optional_cnn_model != None:
        # If we have CNN model, then assess the image with it
        object_ratio = optional_cnn_model.predict(scaled_down_binary_segment_file_name)
    else:
        # Otherwise, we will count the regions and pixels
        temp_img = cv2.imread(scaled_down_image_name)
        temp_width = int(temp_img.shape[1])
        temp_height = int(temp_img.shape[0])
        object_ratio = pixels_count * 100.0 / (temp_height * temp_width)
            
    return (object_ratio, regions_count, scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name)

#------------------------------------------------------
# returns full path to segmented and grey image
# segmented_dir - directory with segmentation 
#                      results
# image_file_name - original path
# step - segmentation step
#------------------------------------------------------
def get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, file_name_suffix, step):

    short_image_file_name = os.path.basename(image_file_name)
    short_image_grey_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_grey_{step}.png").replace("__", "_")
    short_image_binary_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_binary_{step}.png").replace("__", "_")
    
    segmented_grey_file_name = os.path.join(segmented_dir, short_image_grey_file_name)
    segmented_binary_file_name = os.path.join(segmented_dir, short_image_binary_file_name)
    segmented_partially_name_prefix = os.path.join(segmented_dir, short_image_file_name.replace(".png", ""))

    return (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix)

#--------------------------------------------------------
# segment images
# unet - implementation of unet
# labelled_validate_db - path to train db with labels
# labelled_validate_dir - images to segment
# segmented_dir - output dir for segmented images
# step - step size for offsetting windows
# model_path - path from which to load model weights
# batch_size - batch size
# max_labelled_images_count - maximum labelled images,
#                             if -1 then take them all
# padding_to_remove - how many pixels to avoid on border
#--------------------------------------------------------
def segment_image_no_contrast(unet,
                  cml_args,
                  input_full_file_name,
                  input_mask_file_name,
                  segment_dir,
                  sliding_window_step):
  
    # check if segment_dir exists  
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)    
    
    short_image_name = os.path.basename(input_full_file_name)

    (segmented_grey_file_name, segmented_binary_file_name, _) = \
        get_segmented_and_grey_image_file_name(segment_dir, short_image_name, "", sliding_window_step)

    print('input_file_name = {}, output_grey_file_name = {}, output_binary_file_name = {}'.
        format(short_image_name, segmented_grey_file_name, segmented_binary_file_name))

    try:
        unet.segment(cml_args, input_full_file_name, input_mask_file_name, segmented_grey_file_name,
                     segmented_binary_file_name, sliding_window_step)
    except Exception as e:
        print("Exception while segmenting image: {}".format(e))
        (segmented_grey_file_name, segmented_binary_file_name) = (None, None)      
    
    return (segmented_grey_file_name, segmented_binary_file_name)

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def perform_rough_segmentation(unet, 
                               cml_args,
                               temp_dir_name, 
                               input_full_file_name, 
                               rough_sliding_step,
                               min_percent, 
                               max_percent, 
                               percentage_step,
                               optional_cnn_model = None):
    """
    Performs rough segmentation on an input image using a sliding window approach with a large step size and a small step size.
    
    Args:
        unet (keras.models.Model): The trained UNet model to use for segmentation.
        cml_args (argparse.Namespace): The command line arguments to use for segmentation.
        temp_dir_name (str): The name of the temporary directory to use for storing intermediate files.
        input_full_file_name (str): The full path to the input image file.
        rough_sliding_step (int): The sliding step size to use for rough segmentation.
        min_percent (int): The minimum percentage of the image to use for segmentation.
        max_percent (int): The maximum percentage of the image to use for segmentation.
        percentage_step (int): The percentage step size to use for segmentation.
        optional_cnn_model (Cnn): The optional CNN model to use for segmentation quality assessement.
                
    Returns:
        int: The optimal rough segmentation percentage.
    """
    
    # Perform rough segmentation with large step size.
    input_short_file_name = os.path.basename(input_full_file_name)
    file_name_suffix = cml_args.segmentation_suffix
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Rough segmentation with {percentage_step} percent increase]"
    print(log_entry)
    # For single region segmentation we consider average brightness of the most fitting region.
    # For non single region segmentation we consider the area covered by the segmented object
    rough_object_statistics = []
    optimal_rough_object_statistic = 0
    optimal_rough_percentage = min_percent
    eccentricity_level = cml_args.eccentricity_level
    consider_oval_region = cml_args.consider_oval_region
    consider_single_region = cml_args.consider_single_region
    
    for percentage in range(min_percent, max_percent + percentage_step, percentage_step):
        
        try:
            (object_ratio, regions_count, scaled_down_grey_image_name, scaled_down_bw_image_name) = segment_image_by_percentage(
                unet, cml_args, temp_dir_name, input_full_file_name, rough_sliding_step, percentage, optional_cnn_model)
            
            if consider_single_region:   
                brightest_region, brightest_level = get_brightest_region(
                    scaled_down_grey_image_name, scaled_down_bw_image_name, eccentricity_level, consider_oval_region)
                
                if brightest_region is not None:
                    save_brightest_region(brightest_region, scaled_down_grey_image_name)
                    save_brightest_region(brightest_region, scaled_down_bw_image_name)
                rough_object_statistics.append((percentage, brightest_level, 1))                
            else:    
                rough_object_statistics.append((percentage, object_ratio, regions_count))
                
        except Exception as error:
            log_entry = f"[{input_short_file_name}][{file_name_suffix}][Fail][Error: {str(error)}]"
            print(log_entry)
            
    for (percentage, statistic, regions_count) in rough_object_statistics:
        print(f"Percentage: {percentage}, statistic: {statistic}, regions count: {regions_count}")        
        if statistic >= optimal_rough_object_statistic and regions_count == 1:
            # For single region use average brightness to determine optimal percentage
            optimal_rough_object_statistic = statistic
            optimal_rough_percentage = percentage
        
    # For non single region use object ratio to determine optimal percentage    
    if not consider_single_region or (consider_single_region and optimal_rough_percentage == -1):        
        for (percentage, object_ratio, regions_count) in rough_object_statistics:
            if object_ratio >= optimal_rough_object_statistic and regions_count > 0:
                optimal_rough_object_statistic = object_ratio
                optimal_rough_percentage = percentage
    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Out of rough segmentation with step: {percentage_step}, chose percentage: {optimal_rough_percentage}, and optimal statistic: {optimal_rough_object_statistic}, single region: {consider_single_region}]"
    print(log_entry)
  
    return optimal_rough_percentage
