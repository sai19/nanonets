# Face similarity project

This is a re-implementation of facenet paper. The starter code is available at https://github.com/davidsandberg/facenet

## Config 
Config.py contains all the meta information about the training/testing (e.g. batch_size, learning rate, pretrained_model selection, output_path etc.). Use this file to configure the training and do experiments. Config.py is initialized with the parameters given in starter code. Important attributes in config are listed below: <br />
**data_dir** (root_dir for training data)<br />
**model_base_dir** (path to save checkpoints) <br />
**log_base_dir** (path to save training logs) <br />
**embedding_size** (face embedding dimension, default=512) <br />
**distance_metric** (which metric to use, euclidean or cosine similarity)<br />


## Preprocessing step
As a preprocessing step, we extract the face from a given image. There are two available options for face detection **"HARR"** and **"MTCNN"**. you can chose this by modifying config.py. The preprocessing step may take a while depending on the data size.

## Training, predicting and testing
A trained model that achieves an accuracy of 0.937 is provided (20181104-191806/)<br />
Modify the config.py file, and start the training, inference, validation using following commands:<br />
**python train.py**<br />
<br />
**python predict.py --input_path_a path\to\input_a --input_path_b path\to\input_b --multiple_pairs "false" --out_json /path/to/save/predictions**<br />
<br />
**python validate_on_lfw.py**<br />

## Directory structure for training
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
Once the configuration is done, the training can be started.
## Data augmentation
Standard data augmentation such as rotation, flip, crop are used. You can adjust the flag of these augmentation from config file.
## Prediction module
Prediction module expects four parameters: input_a, input_b, multiple_pairs, out_json. The default values of multiple_pairs and out_json are "false" and "prediction.json" respectively. If multiple_pairs is true then input_a, input_b should be csv files with the path info of individual images (sample csv files are provided). The output is the similarity score of all the corrosponding pairs.

## Validation module
validate_on_lfw is forked as it is from the starter code. This is used to evaluate the performance of the network. The current version achieves an accuracy 0.937, with arccos as the similarity measure between the embedding. The similarity score can be changed by updating config.py 
