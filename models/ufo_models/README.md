Intro
The folder contains the graph(.pb), the output_labels.txt and the checkpoint files
Using Shape as the classifying parameter for retraining objects we have utilised the following 12 shapes:
Blimp, cigar, circle, cylinder, disc, fireball, oval, saturn-like, sphere, square, star-like, triangle. 
we extracted unique images per shape and then out of the bunch 12 shape files had more images than 30 (which is required for the internal partition for train,test and verification stages)

Steps for the retraining of the Inception V3 model for the objects:
Install tensorflow Hub
$ pip install "tensorflow>=1.7.0"
$ pip install tensorflow-hub

Activate tensorflow
source ~/tensorflow/bin/activate # bash, sh, ksh, or zsh

Generate the input folder containing the sub-folders used for retraining
ufostalkerimages (the images were resized wrt aspect ratio)

Run the retrainer script(https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py)
python retrain.py --image_dir ~/path_to_ufostalkerimages

This generates the retrained model that can be visualized on the Tensorboard

Steps to deploy the retrained model using the label_image script:(https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py)

python label_image.py \
--graph=/tmp/output_graph.pb 				//graph that contains the retrained model
--labels=/tmp/output_labels.txt \			//output labels are the shapes like blimp,cigar,circle etc.
--input_layer=Placeholder \
--output_layer=final_result \
--image=path_to_image_to_be_classified

 Following is a sample input and output snippet:

(tensorflow) sayali@sayali-VirtualBox:~/Desktop/example_code$ python label_image.py \
 --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
 --input_layer=Placeholder \
 --output_layer=final_result \
 --image=$HOME/Desktop/ufostalkerimages/Blimp/82805.jpg
2018-04-12 02:33:51.419390: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
blimp 0.93306917
saturn like 0.012200303
sphere 0.010724315
fireball 0.009860759
triangle 0.00829932



Integration with the current docker file:
The tika-dockers repo contains the InceptionRestDockerFile that holds the current training model namely Inception V4.
The Path to inception_v4.ckpt & meta files needs to be replaced with the custom paths to our retrained model.ckpt and meta files.


Training with the MobileNet_V2 architecture for higher speeds and deployment on mobile and IOT devices.


