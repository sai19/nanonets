# Face similarity project

This is a re-implementation of facenet paper. The starter code is available at https://github.com/davidsandberg/facenet

## Using the module
A pretrained model that achieves an accuracy of 0.937 is provided (20181104-191806/)<br />
Modify the config.py file, and start the training, inference, validation using following commands:<br />
python train.py<br />
python predict.py<br />
python validate_on_lfw.py<br />

> Folder structure options and naming conventions for software projects
## Using the module
Expected training image directory structure
'''
├── person1                          
    ├── person1_1.jpg 
    ├── person1_2.jpg                   
├── person2                
  ├── person2_1.jpg 
  ├── person2_2.jpg 
'''
