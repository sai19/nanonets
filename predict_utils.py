'''
evaluation script for similarity score between images
configure the config.py file and run the script with following command
python predict.py --image_path_a /path_image_a/ --image_path_b /path_image_b/
Modified by Saiprasad Koturwar (original code at https://github.com/davidsandberg/facenet)
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
from config import Config
import json 
import pandas as pd
import subprocess

def preprocess(config_,input_path_a,input_path_b,multiple_pairs):
    if config_.preprocessing_type_test is not None:
        if config_.preprocessing_type_test=="MTCNN":
            argv = ['python',
                    './align/align_images_mtcnn.py',
                     '--input_path_a', input_path_a,
                     '--input_path_b', input_path_b,
                     '--multiple_pairs', multiple_pairs,
                     '--output_dir', config_.preprocessed_out_dir_test,
                    '--image_size', '160',
                    '--margin', '44']
            print("cleaning the images using MTCNN... This may take time depending on input size")
            subprocess.call(argv)
            print("Done..")
        elif config_.preprocessing_type_test=="HARR":
            argv = ['python',
                    'harr_images.py',
                     input_path_a,
                     input_path_b,
                     config_.preprocessed_out_dir_test]
            print("cleaning the images using HARR... This may take time depending on input size")
            subprocess.call(argv)
            print("Done")
        else:
            print("skipping preprocessing") 

def get_similarity(config_,input_path_a,input_path_b,multiple_pairs=False):
    config_.display()
    # input image size, can be changed in config.py
    image_size = (config_.im_height, config_.im_width)
    if not multiple_pairs:
        # preprocess the input images
        preprocess(config_,input_path_a,input_path_b)
        # get the updated values of image names
        tf.reset_default_graph()
        image_path_a,ext_a = os.path.splitext(os.path.split(input_path_a)[1])
        image_path_b,ext_b = os.path.splitext(os.path.split(input_path_b)[1])
        image_path_a = os.path.join(config_.preprocessed_out_dir_test, image_path_a+'_cropped'+ext_a) 
        image_path_b = os.path.join(config_.preprocessed_out_dir_test, image_path_b+'_cropped'+ext_b)   
        # create the paths for inference
        paths, actual_issame = [(image_path_a,image_path_b),(image_path_a,image_path_b),(image_path_a,image_path_b)],[False]*3
    if multiple_pairs:
        input_path_a_list = list(pd.read_csv(input_path_a)["paths"])
        input_path_b_list = list(pd.read_csv(input_path_b)["paths"])
        preprocess(config_,input_path_a,input_path_b,'true')
        paths,actual_issame = [],[] 
        for i in range(len(input_path_a_list)):
            input_path_a, input_path_b = input_path_a_list[i], input_path_b_list[i]
            image_path_a,ext_a = os.path.splitext(os.path.split(input_path_a)[1])
            image_path_b,ext_b = os.path.splitext(os.path.split(input_path_b)[1])
            image_path_a = os.path.join(config_.preprocessed_out_dir_test, image_path_a+'_cropped'+ext_a) 
            image_path_b = os.path.join(config_.preprocessed_out_dir_test, image_path_b+'_cropped'+ext_b)
            paths.append((image_path_a,image_path_b))
            actual_issame.append(False) # dummy values 

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
            facenet.load_model(config_.pretrained_model, input_map=input_map)

            # Get output tensor, needed to create inference graph
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(coord=coord, sess=sess)
            
            similarity = evaluate(sess, eval_enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
                embeddings, label_batch, paths, actual_issame,config_.pred_batch_size)
            # write the output to json
            # similarity threshold calculated based on validation performance
            are_same = ["Yes" if dist<0.38 else "No" for dist in similarity]
            similarity = [str(dist) for dist in similarity]
            out_data = {"similarity_score":similarity,"are_same_people":are_same}
            #save the info to json file
            return similarity, out_data


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
    nrof_embeddings = len(image_paths)*2  # total no. of embeddings
    nrof_images = nrof_embeddings * 1
    labels_array = np.expand_dims(np.arange(0,nrof_images),1) # indexed array 
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),1),1) # image_paths
    control_array = np.zeros_like(labels_array, np.int32) # augmentation_info
    # run the session to read the images and perform augmentation(if any) 
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    # get the embedding of corrosponding images
    embedding_size = int(embeddings.get_shape()[1])
    if nrof_images>100:
        batch_size = 100
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        emb_array[lab, :] = emb
    embeddings1 = emb_array[0::2]
    embeddings2 = emb_array[1::2]     
    # get the cosine similarity
    cosine_similarity = distance(embeddings1,embeddings2,1)
    return cosine_similarity
