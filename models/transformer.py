import logging
import random
import re

import tensorflow as tf
from tensorflow.python.ops import init_ops

import third_party.tensor2tensor.common_attention as common_attention
import third_party.tensor2tensor.common_layers as common_layers
from utils import average_gradients, shift_right, embedding, residual, dense, ff_hidden, shift_right_latent
from utils import learning_rate_decay, multihead_attention
from model import Model

class Transformer(Model):
    def __init__(self, *args, **kargs):
        super(Transformer, self).__init__(*args, **kargs)
        activations = {"relu": tf.nn.relu,
                       "sigmoid": tf.sigmoid,
                       "tanh": tf.tanh,
                       "swish": lambda x: x * tf.sigmoid(x),
                       "glu": lambda x, y: x * tf.sigmoid(y)}
        self._ff_activation = activations[self._config.ff_activation]

    def encoder_impl(self, encoder_input, is_training):

        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        # Mask
        encoder_padding = tf.equal(encoder_input, 0)
        # Embedding
        encoder_output = embedding(encoder_input,
                                   vocab_size=self._config.src_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   multiplier=self._config.hidden_units**0.5 if self._config.scale_embedding else 1.0,
                                   name="src_embedding")
        # Add positional signal
        encoder_output = common_attention.add_timing_signal_1d(encoder_output)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)
        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=common_attention.attention_bias_ignore_padding(encoder_padding),
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              name='encoder_self_attention',
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)
                # Feed Forward
                encoder_output = residual(encoder_output,
                                          ff_hidden(
                                              inputs=encoder_output,
                                              hidden_size=4 * self._config.hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)

        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
        return encoder_output

    def decoder_impl(self, decoder_input, encoder_output, is_training):

        """
        :param decoder_input:  2D Tensors [batch_size, dst_length]
        :param encoder_output: 3D Tensors [batch_size, src_length, hidden_units]
        :param is_training:
        :return: decoder_output: 3D Tensors [batch_size, dst_length, hidden_units]
        """

        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)

        # Bias for preventing peeping later information
        self_attention_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[1])

        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=self_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              output_depth=self._config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self._config.hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)
                #decoder_output = tf.sigmoid(decoder_output)
        return decoder_output

    def decoder_label_impl(self, decoder_input, encoder_output, is_training):

        """
        :param decoder_input:  2D Tensors [batch_size, dst_length]
        :param encoder_output: 3D Tensors [batch_size, src_length, hidden_units]
        :param is_training:
        :return: decoder_output: 3D Tensors [batch_size, dst_length, hidden_units]
        """

        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.lbl_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="lbl_embedding")
        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)

        # Bias for preventing peeping later information
        self_attention_bias = common_attention.attention_bias_lower_triangle(tf.shape(decoder_input)[1])

        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=self_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              output_depth=self._config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                decoder_output_label = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self._config.hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)

                decoder_output_label = tf.sigmoid(decoder_output_label)

        return decoder_output_label

    def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training):

        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)

        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.dst_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="dst_embedding")
        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)

        new_cache = []

        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output[:, -1:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=None,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              reserve_last=True,
                                              output_depth=self._config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              reserve_last=True,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self._config.hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)

        return decoder_output, new_cache

    def decoder_label_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, is_training):

        attention_dropout_rate = self._config.attention_dropout_rate if is_training else 0.0
        residual_dropout_rate = self._config.residual_dropout_rate if is_training else 0.0

        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = common_attention.attention_bias_ignore_padding(encoder_padding)



        decoder_output = embedding(decoder_input,
                                   vocab_size=self._config.lbl_vocab_size,
                                   dense_size=self._config.hidden_units,
                                   multiplier=self._config.hidden_units ** 0.5 if self._config.scale_embedding else 1.0,
                                   name="lbl_embedding")
        # Positional Encoding
        decoder_output += common_attention.add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=residual_dropout_rate,
                                           training=is_training)

        new_cache = []

        # Blocks
        for i in range(self._config.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output[:, -1:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=None,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              reserve_last=True,
                                              output_depth=self._config.hidden_units,
                                              name="decoder_self_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self._config.hidden_units,
                                              total_value_depth=self._config.hidden_units,
                                              output_depth=self._config.hidden_units,
                                              num_heads=self._config.num_heads,
                                              dropout_rate=attention_dropout_rate,
                                              reserve_last=True,
                                              name="decoder_vanilla_attention",
                                              summaries=True),
                                          dropout_rate=residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self._config.hidden_units,
                                              output_size=self._config.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=residual_dropout_rate)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)

        return decoder_output, new_cache