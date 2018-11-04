# Face similarity project

This is a re-implementation of facenet paper. The starter code is available at https://github.com/davidsandberg/facenet

## Using the module
A trained model that achieves an accuracy of 0.937 is provided (20181104-191806/)<br />
Modify the config.py file, and start the training, inference, validation using following commands:<br />
python train.py<br />
python predict.py<br />
python validate_on_lfw.py<br />

## Using the module
Expected training image directory structure (i.e. every folder should contain the images of a unique person), pass the root path by modifying config.py 
```bash
root
├── person1
│   ├── person1_1
│   ├── person1_2
├── person2
│   ├── person2_1
│   ├── person2_2
```
