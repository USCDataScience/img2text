# Retrain Inception v4 Model (Image Recognition Deep Learning model)

The model in this folder has been retrained with the UFO images. The based model is from the Image Recognition Deep Learning model Project (UFO object types) that used the Inception v4 Model.

This folder includes model checkpoint files and a freeze graph of the model.
The model checkpoint files can be used for further retraining.

The graph (.pb) can be used to classify UFO images into five categories, i.e. UFO, Flare, Dust, Bird, Plane.

## How to obtain the checkpoint and the graph (.pb) file?
```
sh extract.sh
```

## How to integrate with Tika-Docker (USCDatascience)?
Please refers to https://wiki.apache.org/tika/TikaAndVision
for downloading tika-docker and required config files.

After downloading the config files. You have to update the "InceptionRestDockerfile" to apply the retrain checkpoint files to the tika-docker.

Below is the old code that we need to update
```
RUN curl -O http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz && \
    tar -xzvf inception_v4_2016_09_09.tar.gz && rm -rf inception_v4_2016_09_09.tar.gz && \
```

The docker file download the inceptionv4 model checkpoint file.
However, We want to use the retrain inceptionv4 checkpoint instead so we use command
```
COPY path/to/each/checkpoint/file /usr/share/apache-tika/models/dl/image-video/recognition/inceptionv4.ckpt
```
This command will copy the checkpoint file to the docker container path.

## How to use the graph model (.pb)?
You can use the "label_image.py" file to validate the model like so,

```
python label_image.py --graph=path/to/output_graphv4.pb --labels=path/to/output_labelsv4.txt --image=path/to/image/file --input_height=299 --input_width=299 --input_mean=128 --input_std=128 --input_layer=InputImage --output_layer=final_result
```