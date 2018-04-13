# Steps:

This folder contains output_labels.txt and output_graph.pb

THis model has retrained on classes of UFO sightings:
  * spotlight
  * roundworm
  * nematode
  
  
We scraped images from ufostalker website. We then found 3 most occuring classes. We wanted to have more images per class.

We retrain an already existing model based on our images. the steps for those are found here:    
        https://www.tensorflow.org/tutorials/image_retraining 
        
Requirements: tensorflow

## Steps for UFO sightings:
  1. Create directory structure such that images of given class are in single folder. all these folders are within main folder.
  
  2. Next run the steps mentioned in tutorial using retraining.py
  
  3. This will give you output_labels.txt and output_graph.pb
  
  After this you can use the model to test on a new image. Sample command is:
  python label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=<test_image_path>
