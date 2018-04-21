# Steps:
The folder contains the graph(.pb), the output_labels.txt and the checkpoint files
Added a UFO classifier model that classifies UFO images into 5 classes viz. Celestial Objects, Stars, Aeroplane, Glider and Air Balloon.
The aim of this model is to classify the UFO in the image into one of the object classes that are commonly mistaken for UFOs.
For each class of object, we have about 100 images each that we downloaded from imageNET. We have used 100 sample images for each class in order to train the model.
**Steps for the retraining of the Inception V3 model for the objects:**

1   Install tensorflow Hub
````
$ pip install "tensorflow>=1.7.0"
$ pip install tensorflow-hub
````
2   Activate tensorflow
````
source ~/tensorflow/bin/activate # bash, sh, ksh, or zsh
````
3   Generate the input folder containing the sub-folders used for retraining.

4   Run the retrainer script:
(https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py)
````
python retrain.py --image_dir ~/path_to_ufostalkerimages
````
5   This generates the retrained model that can be visualized on the Tensorboard.

**Steps to deploy the retrained model using the label_image script:**

(https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py)

````
python label_image.py \
--graph=/tmp/output_graph.pb 				//graph that contains the retrained model
--labels=/tmp/output_labels.txt \			//output labels are the classes Celestial Objects, Stars, Aeroplane, Glider and Air Balloon.
--input_layer=Placeholder \
--output_layer=final_result \
--image=path_to_image_to_be_classified
**Integration with the current docker file:**
The tika-dockers repo contains the InceptionRestDockerFile that holds the current training model namely Inception V4.
The Path to inception_v4.ckpt & meta files needs to be replaced with the custom paths to our retrained model and our checkpoint file can be used.
The checkpoints are located in the folder by the same name

