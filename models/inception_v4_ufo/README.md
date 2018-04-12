# Retrain Inception v4 Model (Image Recognition Deep Learning model)

The model in this folder has been retrained with the UFO images. The based model is from the Image Recognition Deep Learning model Project (UFO object types) that used the Inception v4 Model.

This folder includes model checkpoint files and a freeze graph of the model.
The model checkpoint files can be used for further retraining.

The graph (.pb) can be used to classify UFO images into five categories, i.e. UFO, Flare, Dust, Bird, Plane.

## How to obtain the checkpoint and the graph (.pb) file?
```
sh extract.sh
```