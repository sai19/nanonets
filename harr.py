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
	root_dir = args.root_dir
	files = os.listdir(root_dir)
	dest_dir = args.dest_dir
	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)
	for person in files:
		person_image_path = root_dir + person
		person_image_path_out = dest_dir+person 
		if not os.path.exists(person_image_path_out):
			os.mkdir(person_image_path_out)
		for person_img in os.listdir(person_image_path):
			cropped_face = crop_face(person_image_path+person_img)
			out_name = person_img.split(".")[0] + _"cropped_" + person_img.split(".")[-1]
			cv2.imwrite(person_image_path_out+out_name,cropped_face)
			


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, 
        help='root_dir', default='')
    parser.add_argument('--dest_dir', type=str, 
        help='path to destination', default='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))