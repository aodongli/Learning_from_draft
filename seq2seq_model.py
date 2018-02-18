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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

# from tensorflow.models.rnn.translate import data_utils
import rnn_cell
import data_utils
import seq2seq_al

SEED = 123


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, source_vocab_size_1, source_vocab_size_2, target_vocab_size, buckets,
                 # size, #annotated by yfeng
                 hidden_edim, hidden_units,  # added by yfeng
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor,
                 beam_size,  # added by shiyue
                 constant_emb_en, # added by al
                 constant_emb_fr, # added by al
                 use_lstm=False,
                 num_samples=10240, forward_only=False):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          #size: number of units in each layer of the model.#annotated by yfeng
          hidden_edim: number of dimensions for word embedding
          hidden_units: number of hidden units for each layer
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.source_vocab_size_1 = source_vocab_size_1
        self.source_vocab_size_2 = source_vocab_size_2
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.
        # if num_samples > 0 and num_samples < self.target_vocab_size:
        if num_samples > 0:
            # w = tf.get_variable("proj_w", [size, self.target_vocab_size])  #annotated by feng
            w = tf.get_variable("proj_w", [hidden_units // 2, self.target_vocab_size],
                                initializer=tf.random_normal_initializer(0, 0.01, seed=SEED))  # added by yfeng
            # w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.target_vocab_size],
                                initializer=tf.constant_initializer(0.0), trainable=False)  # added by yfeng
            output_projection = (w, b)

            def sampled_loss(logit, target):
                # labels = tf.reshape(labels, [-1, 1])
                logit = nn_ops.xw_plus_b(logit, output_projection[0], output_projection[1])
                # return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                #                                   self.target_vocab_size)
                target = array_ops.reshape(target, [-1])
                return nn_ops.sparse_softmax_cross_entropy_with_logits(
                        logit, target)

            softmax_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        # single_cell = tf.nn.rnn_cell.GRUCell(hidden_units) #annotated by yfeng
        single_cell = rnn_cell.GRUCell(hidden_units)  # added by yfeng
        if use_lstm:
            # single_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units) #annotated by yfeng
            single_cell = rnn_cell.BasicLSTMCell(hidden_units)  # added by yfeng
        cell = single_cell
        if num_layers > 1:
            # modified by yfeng
            # cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
            cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
            # end by yfeng
        cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=0.8, seed=SEED)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs, do_decode):
            # return tf.nn.seq2seq.embedding_attention_seq2seq( #annnotated by yfeng
            return seq2seq_al.embedding_attention_seq2seq(  # added by yfeng
                    encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2,
                    decoder_inputs, cell,
                    num_encoder_symbols_1=source_vocab_size_1,
                    num_encoder_symbols_2=source_vocab_size_2,
                    num_decoder_symbols=target_vocab_size,
                    # embedding_size=size,  #annotated by yfeng
                    embedding_size=hidden_edim,  # added by yfeng
                    beam_size=beam_size,  # added by shiyue
                    constant_emb_en=constant_emb_en, # added by al
                    constant_emb_fr=constant_emb_fr, # added by al
                    output_projection=output_projection,
                    feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs_1 = []
        self.encoder_inputs_2 = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs_1.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}_1".format(i)))

        for i in xrange(buckets[-1][1]):  # Last bucket is the biggest one.
            self.encoder_inputs_2.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}_2".format(i)))

        for i in xrange(buckets[-1][2] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))
        self.encoder_mask_1 = tf.placeholder(tf.int32, shape=[None, None],
                                           name="encoder_mask_1")
        self.encoder_mask_2 = tf.placeholder(tf.int32, shape=[None, None],
                                           name="encoder_mask_2")

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            # self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets( #annotated by yfeng
            self.outputs, self.losses, self.symbols = seq2seq_al.model_with_buckets(  # added by yfeng and shiyue
                    self.encoder_inputs_1, self.encoder_inputs_2,
                    self.encoder_mask_1, self.encoder_mask_2,
                    self.decoder_inputs, targets,
                    self.target_weights, buckets, lambda x1, x2, y1, y2, z: seq2seq_f(x1, x2, y1, y2, z, True),
                    softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            # annotated by shiyue, when using beam search, no need to do decoding projection
            # if output_projection is not None:
            #     for b in xrange(len(buckets)):
            #         self.outputs[b] = [
            #             tf.matmul(output, output_projection[0]) + output_projection[1]
            #             for output in self.outputs[b]
            #             ]
            # ended by shiyue
        else:
            # self.outputs, self.losses = tf.nn.seq2seq.model_with_buckets(  #annotated by yfeng
            self.outputs, self.losses, self.symbols = seq2seq_al.model_with_buckets(  # added by yfeng and shiyue
                    self.encoder_inputs_1, self.encoder_inputs_2,
                    self.encoder_mask_1, self.encoder_mask_2,
                    self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x1, x2, y1, y2, z: seq2seq_f(x1, x2, y1, y2, z, False),
                    softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params_to_update = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.gradient_norms_print = []
            self.updates = []
            # opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-6)
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # opt = tf.train.GradientDescentOptimizer(self.learning_rate) #added by yfeng
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params_to_update,
                                         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                # gradients_print = tf.gradients(self.losses[b], params_to_print)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                # _, norm_print = tf.clip_by_global_norm(gradients_print,
                #                                                  max_gradient_norm)
                self.gradient_norms.append(norm)
                # self.gradient_norms_print.append(norm_print)
                self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params_to_update), global_step=self.global_step))

        # self.saver = tf.train.Saver(tf.all_variables()) #annotated by yfeng
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000,
                                    keep_checkpoint_every_n_hours=6)  # added by yfeng

    def step(self, session, encoder_inputs_1, encoder_inputs_2, encoder_mask_1, encoder_mask_2, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size_1, encoder_size_2, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs_1) != encoder_size_1:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs_1), encoder_size_1))
        if len(encoder_inputs_2) != encoder_size_2:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs_2), encoder_size_2))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size_1):
            input_feed[self.encoder_inputs_1[l].name] = encoder_inputs_1[l]
        for l in xrange(encoder_size_2):
            input_feed[self.encoder_inputs_2[l].name] = encoder_inputs_2[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        input_feed[self.encoder_mask_1.name] = encoder_mask_1
        input_feed[self.encoder_mask_2.name] = encoder_mask_2

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            # modified by shiyue
            if self.symbols[0]:
                for l in xrange(decoder_size):  # Output symbols
                    output_feed.append(self.symbols[bucket_id][l])
            else:
                for l in xrange(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size_1, encoder_size_2, decoder_size = self.buckets[bucket_id]
        encoder_inputs_1, encoder_inputs_2, decoder_inputs = [], [], []
        encoder_mask_1, encoder_mask_2 = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input_1, encoder_input_2, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad_1 = [data_utils.PAD_ID] * (encoder_size_1 - len(encoder_input_1))
            encoder_pad_2 = [data_utils.PAD_ID] * (encoder_size_2 - len(encoder_input_2))
            encoder_inputs_1.append(list(encoder_input_1 + encoder_pad_1))
            encoder_inputs_2.append(list(encoder_input_2 + encoder_pad_2))
            encoder_mask_1.append([1] * len(encoder_input_1) + [0] * (encoder_size_1 - len(encoder_input_1)))
            encoder_mask_2.append([1] * len(encoder_input_2) + [0] * (encoder_size_2 - len(encoder_input_2)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs_1, batch_encoder_inputs_2, batch_decoder_inputs, batch_weights = [], [], [], []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size_1):
            batch_encoder_inputs_1.append(
                    np.array([encoder_inputs_1[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=np.int32))
        for length_idx in xrange(encoder_size_2):
            batch_encoder_inputs_2.append(
                    np.array([encoder_inputs_2[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs_1, batch_encoder_inputs_2, encoder_mask_1, encoder_mask_2, batch_decoder_inputs, batch_weights
