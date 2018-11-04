from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from predict_utils import get_similarity
import json
import sys
import argparse
from config import Config

def main(args):
	config_ = Config()
	input_path_a = args.input_path_a
	input_path_b = args.input_path_b
	out_json = args.out_json
	multiple_pairs = args.multiple_pairs
	if multiple_pairs=='false':
		multiple_pairs = False
	else:
		multiple_pairs = True	
	similarity, json_data = get_similarity(config_,input_path_a,input_path_b,multiple_pairs)
	with open(out_json,"w") as outfile:
		json.dump(json_data,outfile)
	if len(similarity)==1:	
		print("The similarity score of the given images is: {}".format(float(similarity[0])))
	print("Predictions writtent to {}".format(out_json))	


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_a', type=str, 
        help='path to input image a', default='')
    parser.add_argument('--input_path_b', type=str, 
        help='path to input image b', default='')
    parser.add_argument('--out_json', type=str, 
        help='path to output file', default='predictions.json')
    parser.add_argument('--multiple_pairs', type=str, 
        help='path to output file', default="false")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))