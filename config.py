"""
config file for facenet training and inference
Written by Saiprasad Koturwar
"""
from __future__ import division
import numpy as np


# Default model configuration, equivalent to tensorflows default config files provided in object detection zoo

class Config(object):
	# training parameters
    gpu_memory_fraction = 1.0
    data_dir = '../lfw/'
    lfw_dir = data_dir
    lfw_pairs = 'pairs.txt'
    model_def = 'models.inception_resnet_v1'
    pretrained_model = '20180402-114759/model-20180402-114759.ckpt-275'
    logs_base_dir = ''
    models_base_dir = ''
    max_nrof_epochs = 100
    batch_size = 2
    im_height = 160
    im_width = 160
    epoch_size = 100# usually = len(training_data)/batch_size
    embedding_size = 128 # dimensionality of the layer before softmax 
    aug_crop = True
    aug_flip = True
    aug_rot = True
    img_standardize = True
    keep_probability = True
    dropout_prob = 1.0 # dropout probability
    weight_decay = 0.0 # l2 regularization decay
    log_histograms = True
    center_loss_factor = 0.0 # refer to the paper
    center_loss_alfa = 0.95 # refer to the paper
    prelogits_norm_loss_factor = 0.0
    prelogits_norm_p = 1.0
    prelogits_hist_max = 10.0
    
    # learning rate scheduling
    optimizer = 'ADAGRAD'
    learning_rate = 0.1
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
    val_dir = ''
    val_batch_size = 100

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
