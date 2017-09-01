# img2text
This project contains a [Tensorflow](http://tensorflow.org/) trained model that implements the [GSOC 2017 project Tensorflow Image to Text in Apache Tika](https://wiki.apache.org/tika/GSOC/GSoC2017) based on the paper [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555). The model is split into multiple parts, and the associated [Docker image](https://raw.githubusercontent.com/apache/tika/master/tika-parsers/src/main/resources/org/apache/tika/parser/captioning/tf/Im2txtRestDockerfile) in [Apache Tika](http://tika.apache.org/) puts the model back together during the docker process for use in Tika.

How to obtain the checkpoint file?
===================
`git clone https://github.com/USCDataScience/img2text.git`

`cd img2text`

`sudo chmod a+x download-1M_iters_ckpt.sh`

`./download-1M_iters_ckpt.sh`

Questions, comments?
===================
Send them to [Chris A. Mattmann](mailto:chris.a.mattmann@jpl.nasa.gov) or [Thejan Wijesinghe](mailto:thejan.k.wijesinghe@gmail.com).

Contributors
============
* Chris A. Mattmann, JPL & USC
* Thejan Wijesinghe, University of Moratuwa

License
===
This project is licensed under the [Apache License, version 2.0](http://www.apache.org/licenses/LICENSE-2.0).



