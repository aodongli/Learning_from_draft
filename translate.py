# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import pickle as pkl
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# from tensorflow.models.rnn.translate import data_utils    #annotated by yfeng
# from tensorflow.models.rnn.translate import seq2seq_model   #annotated by yfeng
import data_utils  # added by yfeng
import seq2seq_model  # added by yfeng

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 1.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 80,
                            "Batch size to use during training.")
# start by yfeng
tf.app.flags.DEFINE_integer("hidden_units", 500, "Size of hidden units for each layer.")
tf.app.flags.DEFINE_integer("hidden_edim", 310, "the dimension of word embedding.")
# end by yfeng
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size_1", 15000, "Pre-trained English vocabulary size.")
tf.app.flags.DEFINE_integer("en_vocab_size_2", 10000, "Pre-trained English vocabulary size.")
tf.app.flags.DEFINE_integer("fr_vocab_size", 10000, "English vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "../data_iwslt", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../train_iwslt", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 250,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")
# added by yfeng, for decode
tf.app.flags.DEFINE_string("model", "ckpt", "the checkpoint model to load")
# added by shiyue, for beam search
tf.app.flags.DEFINE_integer("beam_size", 5,
                            "The size of beam search. Do greedy search when set this to 1.")
# added by al, for constant embedding
tf.app.flags.DEFINE_string("constant_emb_en_dir", "emb_en", "constant embedding directory")
tf.app.flags.DEFINE_string("constant_emb_fr_dir", "emb_fr", "constant embedding directory")

FLAGS = tf.app.flags.FLAGS

tf.set_random_seed(123)

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
#_buckets = [(10, 10), (20, 20), (30, 30), (40, 40), (51, 51)]


#_buckets = [(5, 10), (10, 15), (20, 25), (45, 50)] # annotated by al
_buckets = [(5, 10, 10), (10, 15, 15), (20, 25, 25), (45, 50, 50)] # added by al


def read_data(source_path_1, source_path_2, target_path, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
      source_path: path to the files with token-ids for the source language.
      target_path: path to the file with token-ids for the target language;
        it must be aligned with the source file: n-th line contains the desired
        output for n-th line from the source_path.
      max_size: maximum number of lines to read, all other will be ignored;
        if 0 or None, data files will be read completely (no limit).

    Returns:
      data_set: a list of length len(_buckets); data_set[n] contains a list of
        (source, target) pairs read from the provided data files that fit
        into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
        len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path_1, mode="r") as source_file_1:
        with tf.gfile.GFile(source_path_2, mode="r") as source_file_2:
            with tf.gfile.GFile(target_path, mode="r") as target_file:
                source_1, source_2, target = source_file_1.readline(), source_file_2.readline(), target_file.readline()
                counter = 0
                while source_1 and source_2 and target and (not max_size or counter < max_size):
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading data line %d" % counter)
                        sys.stdout.flush()
                    source_ids_1 = [int(x) for x in source_1.split()][:50]
                    source_ids_2 = [int(x) for x in source_2.split()][:50]
                    target_ids = [int(x) for x in target.split()][:50]
                    source_ids_1.append(data_utils.EOS_ID)
                    source_ids_2.append(data_utils.EOS_ID)
                    target_ids.append(data_utils.EOS_ID)
                    for bucket_id, (source_size_1, source_size_2, target_size) in enumerate(_buckets):
                        if len(source_ids_1) < source_size_1 and len(source_ids_2) < source_size_2 and len(target_ids) < target_size:
                            data_set[bucket_id].append([source_ids_1, source_ids_2, target_ids])
                            break
                    source_1, source_2, target = source_file_1.readline(), source_file_2.readline(), target_file.readline()
    return data_set


'''
#annotated by yfeng
def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.en_vocab_size, FLAGS.fr_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model
#end by yfeng
'''


# start by yfeng
def create_model(session,
                 forward_only,
                 ckpt_file=None):
    """Create translation model and initialize or load parameters in session."""
    emb_en_file = file(FLAGS.constant_emb_en_dir, "rb")
    emb_fr_file = file(FLAGS.constant_emb_fr_dir, "rb")
    constant_emb_en = pkl.load(emb_en_file) # added by al
    constant_emb_fr = pkl.load(emb_fr_file) # added by al
    model = seq2seq_model.Seq2SeqModel(
            FLAGS.en_vocab_size_1, FLAGS.en_vocab_size_2, FLAGS.fr_vocab_size, _buckets,
            FLAGS.hidden_edim, FLAGS.hidden_units,  # by yfeng
            FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
            FLAGS.beam_size,  # added by shiyue
            constant_emb_en=constant_emb_en, # added by al
            constant_emb_fr=constant_emb_fr, # added by al
            forward_only=forward_only)
    if ckpt_file:
        model_path = os.path.join(FLAGS.train_dir, ckpt_file)
        if tf.gfile.Exists(model_path):
            sys.stderr.write("Reading model parameters from %s\n" % model_path)
            sys.stderr.flush()
            model.saver.restore(session, model_path)
    else:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            session.run(tf.initialize_all_variables())
    return model


# end by yfeng



def train():
    """Train a en->fr translation model using WMT data."""
    # Prepare WMT data.
    # print("Preparing WMT data in %s" % FLAGS.data_dir)  #annotated by yfeng
    print("Preparing training and dev data in %s" % FLAGS.data_dir)  # added by yfeng
    en_train_1, en_train_2, fr_train, en_dev_1, en_dev_2, fr_dev, en_vocab_path_1, en_vocab_path_2, fr_vocab_path = data_utils.prepare_wmt_data(
            FLAGS.data_dir, FLAGS.en_vocab_size_1, FLAGS.en_vocab_size_2, FLAGS.fr_vocab_size)

    en_vocab_1, rev_en_vocab_1 = data_utils.initialize_vocabulary(en_vocab_path_1)
    en_vocab_2, rev_en_vocab_2 = data_utils.initialize_vocabulary(en_vocab_path_2)
    fr_vocab, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

    if FLAGS.en_vocab_size_1 > len(en_vocab_1):
        FLAGS.en_vocab_size_1 = len(en_vocab_1)
    if FLAGS.en_vocab_size_2 > len(en_vocab_2):
        FLAGS.en_vocab_size_2 = len(en_vocab_2)
    if FLAGS.fr_vocab_size > len(fr_vocab):
        FLAGS.fr_vocab_size = len(fr_vocab)

    with tf.Session() as sess:
        # Create model.
        # print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size)) #annotated by yfeng
        print("Creating %d layers of %d units with word embedding %d."
              % (FLAGS.num_layers, FLAGS.hidden_units, FLAGS.hidden_edim))  # added by yfeng
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)
        dev_set = read_data(en_dev_1, en_dev_2, fr_dev)
        train_set = read_data(en_train_1, en_train_2, fr_train, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs, target_weights = model.get_batch(
                    train_set, bucket_id)

            _, step_loss, _ = model.step(sess, encoder_inputs_1, encoder_inputs_2,
                                         encoder_mask_1, encoder_mask_2,
                                         decoder_inputs,
                                         target_weights, bucket_id, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.8f step-time %.2f perplexity "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, perplexity))

                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]) and model.learning_rate.eval() > 1e-12:
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in xrange(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs, target_weights = model.get_batch(
                            dev_set, bucket_id)
                    _, eval_loss, _ = model.step(sess, encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs,
                                                 target_weights, bucket_id, True)
                    eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))  # annotated by yfeng

                sys.stdout.flush()


def decode():
    with tf.Session() as sess:
        #_buckets = [(10, 10), (20, 20), (30, 30), (40, 40), (51, 51), (100, 100)]
        # Load vocabularies.
        en_vocab_path_1 = os.path.join(FLAGS.data_dir,
                                     "vocab%d.en_1" % FLAGS.en_vocab_size_1)
        en_vocab_path_2 = os.path.join(FLAGS.data_dir,
                                     "vocab%d.en_2" % FLAGS.en_vocab_size_2)
        fr_vocab_path = os.path.join(FLAGS.data_dir,
                                     "vocab%d.fr" % FLAGS.fr_vocab_size)
        en_vocab_1, rev_en_vocab_1 = data_utils.initialize_vocabulary(en_vocab_path_1)
        en_vocab_2, rev_en_vocab_2 = data_utils.initialize_vocabulary(en_vocab_path_2)
        fr_vocab, rev_fr_vocab = data_utils.initialize_vocabulary(fr_vocab_path)

        if FLAGS.en_vocab_size_1 > len(en_vocab_1):
            FLAGS.en_vocab_size_1 = len(en_vocab_1)
        if FLAGS.en_vocab_size_2 > len(en_vocab_2):
            FLAGS.en_vocab_size_2 = len(en_vocab_2)
        if FLAGS.fr_vocab_size > len(fr_vocab):
            FLAGS.fr_vocab_size = len(fr_vocab)

        # Create model and load parameters.
        model = create_model(sess, True, FLAGS.model)
        model.batch_size = 1  # We decode one sentence at a time.

        # Decode from standard input.
        # sys.stdout.write("> ")
        # sys.stdout.flush()
        sentence_1 = sys.stdin.readline()
        sentence_2 = sys.stdin.readline()
        while sentence_1 and sentence_2:
            # Get token-ids for the input sentence.
            token_ids_1 = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence_1), en_vocab_1)
            token_ids_2 = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence_2), en_vocab_2)
            if len(token_ids_1) > _buckets[-1][0]: # Added by al to cut short overlength input
                token_ids_1 = token_ids_1[:_buckets[-1][0]]
            if len(token_ids_2) > _buckets[-1][1]: # Added by al to cut short overlength input
                token_ids_2 = token_ids_2[:_buckets[-1][1]]
            # Which bucket does it belong to?
            bucket_id = [b for b in xrange(len(_buckets))
                         if _buckets[b][0] > len(token_ids_1) and _buckets[b][1] > len(token_ids_2)]
            if bucket_id:
                bucket_id = min(bucket_id)
            else:
                bucket_id = len(_buckets) - 1
            # Get a 1-element batch to feed the sentence to the model.
            encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs, target_weights = model.get_batch(
                    {bucket_id: [(token_ids_1, token_ids_2, [])]}, bucket_id)
            # Get output logits for the sentence.
            _, _, output_logits = model.step(sess, encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs,
                                             target_weights, bucket_id, True)

            # This is a greedy decoder - outputs are just argmaxes of output_logits.
            # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]  # annotated by shiyue

            # This is a beam search decoder - output is the best result from beam search
            outputs = [int(logit) for logit in output_logits]  # added by shiyue

            # If there is an EOS symbol in outputs, cut them at that point.
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            # Print out French sentence corresponding to outputs.
            # sys.stdout.flush()
            print(" ".join([tf.compat.as_str(rev_fr_vocab[output]) for output in outputs]))
            # print("> ", end="")
            sys.stdout.flush()
            sentence_1 = sys.stdin.readline()
            sentence_2 = sys.stdin.readline()


def self_test():
    """Test the translation model."""
    with tf.Session() as sess:
        print("Self-test for neural translation model.")
        # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
        model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                           5.0, 32, 0.3, 0.99, num_samples=8)
        sess.run(tf.initialize_all_variables())

        # Fake data set for both the (3, 3) and (6, 6) bucket.
        data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                    [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
        for _ in xrange(5):  # Train the fake model for 5 steps.
            bucket_id = random.choice([0, 1])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                    data_set, bucket_id)
            model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                       bucket_id, False)


def main(_):
    if FLAGS.self_test:
        self_test()
    elif FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
