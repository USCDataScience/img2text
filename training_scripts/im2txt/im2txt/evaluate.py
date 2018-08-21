# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluate the model.

This script should be run concurrently with training so that summaries show up
in TensorBoard.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
import io


import numpy as np
import tensorflow as tf


from im2txt import configuration
from im2txt import show_and_tell_model
from im2txt import inference_wrapper
from im2txt.inference_utils import bleu_scorer
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", "", "Directory to write event logs.")
tf.flags.DEFINE_string("log_dir", "", "Directory to write event logs.")
tf.flags.DEFINE_string("image_feed_dir","/home/hamsashwethav/training_scripts/test_imgs",
					   "Image to be fed in the inference placeholder")
tf.flags.DEFINE_integer("eval_interval_secs", 300,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 10132,
                        "Number of examples for evaluation.")

tf.flags.DEFINE_integer("min_global_step", 50,
                        "Minimum global step to run evaluation.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("ref_file_dir","","Directory containing reference sentences file for test images")

tf.logging.set_verbosity(tf.logging.INFO)


def fetch_refs(file_name):
    """This function assumes there are files with names as image file names, and each line contains a reference sentence"""
    """Enhancement could be to have a JSON file, as array image_file_name, ref sentences"""
    
    file_path = os.path.join(FLAGS.ref_file_dir, file_name)
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            refs = f.readlines()
    else:
        return ["these are foxes", "bedrock regolith", "sedimentary rocks"]

    return refs

def run():
  """Runs evaluation in a loop, and logs summaries to TensorBoard."""
  g = tf.Graph()
  with g.as_default():
  # Build the model for evaluation.
    model_config = configuration.ModelConfig()
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, FLAGS.checkpoint_dir)
   
  g.finalize()
  # Create the summary operation and the summary writer.

  # summary_op = tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  # Run a new evaluation run every eval_interval_secs.
  while True:
  	start = time.time()
  	tf.logging.info("Starting evaluation at " + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
	with tf.Session(graph=g) as sess:
		restore_fn(sess)
  		# start = time.time()
        		
		vocab = vocabulary.Vocabulary(FLAGS.vocab_file)
    	        generator = caption_generator.CaptionGenerator(model, vocab, beam_size=1)
        	for filename in os.listdir(FLAGS.image_feed_dir):
			with tf.gfile.GFile(os.path.join(FLAGS.image_feed_dir, filename), "rb") as f:
				print(filename)
				image = f.read()
			captions = generator.beam_search(sess, image)
			print("Captions for image %s:" % os.path.basename(filename))
			# for i, caption in enumerate(captions):
                        caption = captions[0]
			sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
			sentence = " ".join(sentence)
			print("%s (p=%f)" % (sentence, math.exp(caption.logprob)))
	    	        # Run evaluation on the latest checkpoint.
			test = sentence
			refs = fetch_refs(filename)
			bleu_s = bleu_scorer.BleuScorer(test, refs, n=3)
			score, lst = bleu_s.compute_score(option="average")
			print(score[0])
                        score_summary = score[0]
                        
                        # global_step = tf.train.global_step(sess, model.global_step.name)
                        # global_step = tf.train.global_step(sess, "global_step")
                        # summary = tf.Summary(value=[tf.Summary.Value(tag="BLEU Score", simple_value=score_summary)])
                        # summary_writer.add_summary(summary, 0)
                        # summary_writer.flush()
	time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
        if time_to_next_eval > 0:
            time.sleep(time_to_next_eval)


def main(unused_argv):
  assert FLAGS.input_file_pattern, "--input_file_pattern is required"
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
  assert FLAGS.eval_dir, "--eval_dir is required"
  run()


if __name__ == "__main__":
  tf.app.run()
