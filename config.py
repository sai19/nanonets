"""
config file for facenet training and inference
Written by Saiprasad Koturwar
"""
from __future__ import division
import numpy as np
import os


# Default model configuration, equivalent to tensorflows default config files provided in object detection zoo

class Config(object):
    # data preprocessing parameters
    preprocessing_type = None # allowed values [MTCNN,HARR,None]
    preprocessed_out_dir = "" # output will be saved with same structure as in original directory
    # training parameters
    gpu_memory_fraction = 0.7
    data_dir = '../lfw_cleaned/'
    lfw_dir = '../lfw/'
    lfw_pairs = 'pairs.txt'
    model_def = 'models.inception_resnet_v1'
    pretrained_model = '../finetuned_models/20181104-135828/'
    logs_base_dir = '../logs/'
    models_base_dir = '../finetuned_models/'
    max_nrof_epochs = 100
    batch_size = 100
    im_height = 160
    im_width = 160
    epoch_size = 100# usually = len(training_data)/batch_size
    embedding_size = 512 # dimensionality of the layer before softmax 
    random_crop = True
    random_flip = True
    random_rotate = True
    img_standardize = True
    keep_probability = 0.8
    use_fixed_image_standardization = True
    dropout_prob = 1.0 # dropout probability
    weight_decay = 5e-4 # l2 regularization decay
    log_histograms = True
    center_loss_factor = 0.0 # refer to the paper
    center_loss_alfa = 0.95 # refer to the paper
    prelogits_norm_loss_factor = 5e-4
    prelogits_norm_p = 1.0
    prelogits_hist_max = 10.0
    
    # learning rate scheduling
    optimizer = 'ADAM'
    learning_rate = -1
    learning_rate_decay_epochs = 100
    learning_rate_decay_factor = 1.0
    moving_average_decay = 0.9999
    seed = 666
    nrof_preprocess_threads = 4
    learning_rate_schedule_file = 'learning_rate_schedule.txt'
    
    # image filtering
    filter_filename = ''
    filter_percentile = 100.0
    filter_min_nrof_images_per_class = 1
    validate_every_n_epochs = 5
    validation_set_split_ratio = 0.0
    min_nrof_val_images_per_class = 0
 
    # Parameters for validation on LFW
    val_annotation = 'data/pairs.txt'
    validation_set_split_ratio = 0.05
    val_dir = ''
    lfw_batch_size = 100
    lfw_nrof_folds = 2
    lfw_distance_metric = 1
    lfw_subtract_mean = True
    lfw_use_flipped_images = True

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
