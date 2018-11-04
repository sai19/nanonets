'''
this script loads the training module and trains the network based on metadata provided in config.py
Alternatively, one can override metadata before calling train_facenet function
Once the config.py is configured, begin training by running this script
python train.py
Written by Saiprasad Koturwar
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from train_utils import train_facenet
from config import Config


if __name__ == '__main__':
    config_ = Config() # refer to config.py script, config contais meta info for training
    """
    override config values here
    e.g. config.batch_size = 32
    """
    train_facenet(config_)