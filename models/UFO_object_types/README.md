

# Instructions 

In order to test this model you can download a template python script that allows you to specify models and inputs.

First clone the google code labs scripts and cd into the folder

```shell
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2
cd tensorflow-for-poets-2
```

Next, while your inside the tensorflow-for-poets folder you need to run the a label image script as follows

```
python -m scripts.label_image \
    --graph={LOCAITON OF THE MODEL }  \
    --image={LOCATION OF IMAGE TO BE CLASSIFIED} \
    --label={LOCATION OF LABELS FOR CLASSIFICATION}
```

Here is a sample input and what kind of output to expect. 

```Shell
 python -m scripts.label_image  
 --graph=tf_files/object_types__retrained_graph.pb     
 --image=tf_files/training_data/flying_objects/90926_submitter_file3__Nic.UFOzoom2.jpg  
 --labels=tf_files/object_types__retrained_labels.txt 
 
 
Evaluation time (1-image): 0.173s

flying objects 0.996902
spheres 0.0030978536
lights 1.9191765e-07
glowing orbs 7.516825e-11
```

If you want to automate this process for a large batch of inputs please use the temple in the above repository or use the  above shell commands with output redirection. 