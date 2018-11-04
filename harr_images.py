"""
script for preprocessing images for training and testing
python harr_images.py --input_path_a /path_image_a/ --input_path_b /path_image_b/ --out_dir /path_to_out/
Written by Saiprasad Koturwar
"""
import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def crop_face(img_path,face_cascade):
	img = cv2.imread(img_path)
	gray = cv2.imread(img,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	max_area = 0:
	for (x,y,w,h) in faces:
		if x*h>max_area:
			max_area = x*h
			X,Y,W,H = x,y,w,h
	cropped_face = img[X:X+W,Y:Y+H]
	cropped_face = cv2.resize(cropped_face,(160,160))
	return cropped_face			
def main(args):
	input_path_a = args.input_path_a
    input_path_b = args.input_path_b
    dest_dir = os.path.expanduser(args.dest_dir)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
	files = [input_path_a,input_path_b]
	for person_img in [input_path_a,input_path_b]
			cropped_face = crop_face(person_img)
			out_name = person_img.split(".")[0] + _"cropped_" + person_img.split(".")[-1]
			cv2.imwrite(dest_dir+out_name,cropped_face)
			


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, 
        help='root_dir', default='')
    parser.add_argument('--dest_dir', type=str, 
        help='path to destination', default='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))