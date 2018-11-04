
'''
evaluation script for similarity score between images
configure the config.py file and run the script with following command
python predict.py --image_path_a /path_image_a/ --image_path_b /path_image_b/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import utils as facenet
import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from config import Config
import json 

def main(args):
    conf = Config()
    conf.display()
    # input image size, can be changed in config.py
    image_size = (conf.im_height, conf.im_width)
    # Get the paths for the corresponding images
    input_path_a = args.input_path_a
    input_path_b = args.input_path_b   
    paths, actual_issame = [(input_path_a,input_path_b)],[False]
    with tf.Graph().as_default():
        with tf.Session() as sess:
            
            image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
            labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels') # labels of the input pairs 
            batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size') # batch size for evaluation, default =1 
            control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control') # control argument for data augmentation, not needed for testing
            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train') # flag train(True)/test(False)
 
            nrof_preprocess_threads = 4
            eval_input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
                                        dtypes=[tf.string, tf.int32, tf.int32],
                                        shapes=[(1,), (1,), (1,)],
                                        shared_name=None, name=None)
            eval_enqueue_op = eval_input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='eval_enqueue_op')
            # input pipeline, needed to define the input/output signatures
            image_batch, label_batch = facenet.create_input_pipeline(eval_input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
     
            # Load the model, modify the path to pretrained model in config.py
            input_map = {'image_batch': image_batch, 'label_batch': label_batch, 'phase_train': phase_train_placeholder}
            facenet.load_model(conf.pretrained_model, input_map=input_map)

            # Get output tensor, needed to create inference graph
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            
            similarity = evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                embeddings, label_batch, paths, actual_issame,conf.batch_size)
            print("the similarity score of the given images is: {}".format(similarity))
            return similarity


# utility function to get the distance between embeddings              
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / np.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist


# run the inference on given images
def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
        embeddings, labels, image_paths, actual_issame, batch_size):
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on the images')
    
    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(image_paths)*2  # total no. of embeddings
    nrof_images = nrof_embeddings * 1
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),1),1)
    control_array = np.zeros_like(labels_array, np.int32)
    # run the session to read the images and perform augmentation(if any) 
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    # get the embedding of corrosponding images
    feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
    emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
    # get the cosine similarity
    cosine_similarity = distance(emb[0,:].reshape((1,-1)),emb[1,:].reshape((1,-1)),1)
    return cosine_similarity
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_a', type=str, 
        help='path to input image a', default='')
    parser.add_argument('--input_path_b', type=str, 
        help='path to input image b', default='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
