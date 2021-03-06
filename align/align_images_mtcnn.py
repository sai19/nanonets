"""
script for preprocessing images for training and testing
python align_image_mtcnn.py --input_path_a /path_image_a/ --input_path_b /path_image_b/ --out_dir /path_to_out/
Modified by Saiprasad Koturwar (original code at https://github.com/kpzhang93/MTCNN_face_detection_alignment)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import random
from time import sleep
import pandas as pd

def main(args):
    sleep(random.random())
    input_path_a = args.input_path_a
    input_path_b = args.input_path_b
    output_path = os.path.expanduser(args.output_dir)
    multiple_pairs = args.multiple_pairs
    if multiple_pairs == "false":
        multiple_pairs = False
    else:
        multiple_pairs = True    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print('Creating networks and loading parameters')
    
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    if multiple_pairs:
        input_path_a_list = list(pd.read_csv(input_path_a)["paths"])
        input_path_b_list = list(pd.read_csv(input_path_b)["paths"])
    else:
        input_path_a_list = [input_path_a]
        input_path_b_list = [input_path_b]
    # Add a random key to the filename to allow alignment using multiple processes
    input_files = input_path_a_list+input_path_b_list
    for image_path in input_files:
        random_key = np.random.randint(0, high=99999)
        filename,ext = os.path.splitext(os.path.split(image_path)[1])
        output_filename = os.path.join(output_path, filename+ext)
        print(output_filename,image_path)
        if not os.path.exists(output_filename):
            try:
                img = misc.imread(image_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(image_path, e)
                print(errorMessage)
            else:
                if img.ndim<2:
                    print('Unable to align "%s"' % image_path)
                    continue
                if img.ndim == 2:
                    img = facenet.to_rgb(img)
                img = img[:,:,0:3]

                bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                nrof_faces = bounding_boxes.shape[0]
                if nrof_faces>0:
                    det = bounding_boxes[:,0:4]
                    det_arr = []
                    img_size = np.asarray(img.shape)[0:2]
                    if nrof_faces>1:
                        if args.detect_multiple_faces:
                            for i in range(nrof_faces):
                                det_arr.append(np.squeeze(det[i]))
                        else:
                            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                            img_center = img_size / 2
                            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                            det_arr.append(det[index,:])
                    else:
                        det_arr.append(np.squeeze(det))

                    for i, det in enumerate(det_arr):
                        det = np.squeeze(det)
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0]-args.margin/2, 0)
                        bb[1] = np.maximum(det[1]-args.margin/2, 0)
                        bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                        bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                        scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                        filename_base, file_extension = os.path.splitext(output_filename)
                        if args.detect_multiple_faces:
                            output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                        else:
                            output_filename_n = "{}{}".format(filename_base, "_cropped"+file_extension)
                        misc.imsave(output_filename_n, scaled)
            

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_path_a', type=str, help='path to input image a')
    parser.add_argument('--input_path_b', type=str, help='path to input image b')
    parser.add_argument('--output_dir', type=str, help='output_directory')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    parser.add_argument('--multiple_pairs', type=str, 
        help='path to output file', default="false")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
