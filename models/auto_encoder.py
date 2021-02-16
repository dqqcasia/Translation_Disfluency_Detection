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


class AutoEncoder(Model):
    def __init__(self, *args, **kargs):
        super(AutoEncoder, self).__init__(*args, **kargs)
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

    def build_train_model(self, test=True, teacher_model=None, reuse=None):
        """Build model for training. """
        logging.info('Build train model.')
        with tf.device(self._sync_device):
            self.prepare_training()

        with self.graph.as_default():
            loss_list, gv_list = [], []
            cache = {}
            load = dict([(d, 0) for d in self._devices])

            for i, (X, Y, device) in enumerate(zip(self.src_pls, self.dst_pls, self._devices)):

                def daisy_chain_getter(getter, name, *args, **kwargs):
                    """Get a variable and cache in a daisy chain."""
                    device_var_key = (device, name)
                    if device_var_key in cache:
                        # if we have the variable on the correct device, return it.
                        return cache[device_var_key]
                    if name in cache:
                        # if we have it on a different device, copy it from the last device
                        v = tf.identity(cache[name])
                    else:
                        var = getter(name, *args, **kwargs)
                        v = tf.identity(var._ref())  # pylint: disable=protected-access
                    # update the cache
                    cache[name] = v
                    cache[device_var_key] = v
                    return v

                def balanced_device_setter(op):
                    """Balance variables to all devices."""
                    if op.type in {'Variable', 'VariableV2', 'VarHandleOp'}:
                        # return self._sync_device
                        min_load = min(load.values())
                        min_load_devices = [d for d in load if load[d] == min_load]
                        chosen_device = random.choice(min_load_devices)
                        load[chosen_device] += op.outputs[0].get_shape().num_elements()
                        return chosen_device
                    return device

                device_setter = balanced_device_setter

                with tf.variable_scope(tf.get_variable_scope(),
                                       initializer=self._initializer,
                                       custom_getter=daisy_chain_getter,
                                       reuse=reuse):
                    with tf.device(device_setter):
                        logging.info('Build train model on %s.' % device)
                        encoder_output = self.encoder(X, is_training=True, reuse=i > 0 or None)
                        loss = self.train_output(encoder_output, Y, teacher_probs=None, reuse=i > 0 or None)
                        loss_list.append(loss)

                        var_list = tf.trainable_variables()
                        if self._config.train.var_filter:
                            var_list = [v for v in var_list if re.match(self._config.train.var_filter, v.name)]
                        gv_list.append(self._optimizer.compute_gradients(loss, var_list=var_list))

                        logging.info('Build train model on %s finished.' % device)

            self.loss = tf.reduce_mean(loss_list)

            # Clip gradients and then apply.
            grads_and_vars = average_gradients(gv_list)

            if self._config.train.grads_clip > 0:
                grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                                clip_norm=self._config.train.grads_clip)
                grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
            else:
                self.grads_norm = tf.global_norm([gv[0] for gv in grads_and_vars])

            self.train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

            # Summaries
            tf.summary.scalar('loss', loss)
            self.summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver(var_list=tf.global_variables())

        # We may want to test the model during training.
        if test:
            self.build_test_model(reuse=True)

    def build_test_model(self, reuse=None):
        """Build model for inference."""
        logging.info('Build test model.')
        with self.graph.as_default(), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            prediction_list = []
            loss_sum = 0
            for i, (X, Y, device) in enumerate(zip(self.src_pls, self.dst_pls, self._devices)):
                with tf.device(device):
                    logging.info('Build model on %s.' % device)

                    # Avoid errors caused by empty input by a condition phrase.
                    enc_output = self.encoder(X, is_training=False, reuse=i > 0 or None)
                    loss, _, preds = self.test_loss(enc_output, Y, reuse=True)
                    loss_sum += loss
                    prediction_list.append(preds)

            max_length = tf.reduce_max([tf.shape(pred)[1] for pred in prediction_list])

            def pad_to_max_length(input, length):
                """Pad the input (with rank 2) with 3(</S>) to the given length in the second axis."""
                shape = tf.shape(input)
                padding = tf.ones([shape[0], length - shape[1]], dtype=tf.int32) * 3
                return tf.concat([input, padding], axis=1)

            preds_list = [pad_to_max_length(pred, max_length) for pred in prediction_list]
            self.prediction = tf.concat(preds_list, axis=0, name='predictions')
            self.loss_sum = tf.identity(loss_sum, name='loss_sum')
        logging.info('Build test model finished.')

    def train_output(self, encoder_output, Y, teacher_probs, reuse):
        """Calculate loss and accuracy."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            logits = dense(encoder_output, self._config.dst_vocab_size, use_bias=False,
                               name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                               reuse=True if self._config.tie_embedding_and_softmax else None)  # 2D to 3D
            preds = tf.to_int32(tf.argmax(logits, axis=-1))

            mask = tf.to_float(tf.not_equal(Y, 0))

            # Token-level accuracy
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)
            if not tf.get_variable_scope().reuse:
                tf.summary.scalar('accuracy', acc)

            if teacher_probs is not None:
                # Knowledge distillation
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=teacher_probs)
            else:
                # Smoothed loss
                loss = common_layers.smoothing_cross_entropy(logits=logits, labels=Y,
                                                             vocab_size=self._config.dst_vocab_size,
                                                             confidence=1 - self._config.train.label_smoothing)
            loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)

            return loss


    def test_loss(self, encoder_output, Y, reuse):
        """This function help users to compute PPL and predictions during test."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            logits = dense(encoder_output, self._config.dst_vocab_size, use_bias=False,
                           name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                           reuse=True if self._config.tie_embedding_and_softmax else None)
            preds = tf.to_int32(tf.argmax(logits, axis=-1)) # [batch_size, length]
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=self._config.dst_vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_sum = tf.reduce_sum(loss * mask)
            probs = tf.nn.softmax(logits)
        return loss_sum, probs, preds




