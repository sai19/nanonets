import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def main(args):

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, 
        help='root_dir', default='')
    parser.add_argument('--dest_dir', type=str, 
        help='path to destination', default='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))