# Retrain Inception v3 Model (ImageCaptioning)

The model in this folder has been retrained with the UFO images. The based model is from the ImageCaptioning Project that used the Inception v3 Model.

This folder includes model checkpoint files and a freeze graph of the model.
The model checkpoint files can be used for further retraining.

The graph (.pb) can be used to classify UFO images into five categories, i.e. UFO, Flare, Dust, Bird, Plane.

## How to obtain the checkpoint and the graph (.pb) file?
```
sh extract.sh
```

## How to integrate with Tika-Docker (USCDatascience)?
Please refers to https://wiki.apache.org/tika/ImageCaption
for downloading tika-docker and required config files.

After downloading the config files. You have to update the "Im2txtRestDockerfile" to apply the retrain checkpoint files to the tika-docker.

Below is the old code that we need to update
```
RUN echo "We're downloading the checkpoint file for image captioning, the shell might look unresponsive. Please be patient."  && \
    # To get rid of early EOF error
    git config --global http.postBuffer 1048576000 && \
    git clone https://github.com/USCDataScience/img2text.git && \
    # Join the parts
    cat img2text/models/1M_iters_ckpt_parts_* >1M_iters_ckpt.tar.gz && \
    tar -xzvf 1M_iters_ckpt.tar.gz && rm -rf 1M_iters_ckpt.tar.gz
```

The docker file download the inceptionv3 model checkpoint file.
However, We want to use the retrain inceptionv3 checkpoint instead so we use command
```
COPY path/to/each/checkpoint/file /usr/share/apache-tika/models/dl/image/caption/1M_iters_ckpt/model.ckpt-1000000
```
This command will copy the checkpoint file to the docker container path.

## How to use the graph model (.pb)?
You can use the "label_image.py" file to validate the model like so,

```
python label_image.py --graph=path/to/output_graph.pb --labels=path/to/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=path/to/image/file
```