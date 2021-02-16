import logging
import random
import re

import tensorflow as tf
from tensorflow.python.ops import init_ops

import third_party.tensor2tensor.common_attention as common_attention
import third_party.tensor2tensor.common_layers as common_layers
from utils import average_gradients, shift_right, embedding, residual, dense, ff_hidden, shift_right_latent
from utils import learning_rate_decay, multihead_attention


class Model(object):
    def __init__(self, config, num_gpus):
        self.graph = tf.Graph()
        self._config = config

        self._devices = [
            '/gpu:%d' %
            i for i in range(
                int(num_gpus))] if int(num_gpus) > 0 else ['/cpu:0']
        self._sync_device = self._devices[0] if len(
            self._devices) == 1 else '/cpu:0'

        # Placeholders and saver.
        with self.graph.as_default():
            src_pls = []
            dst_pls = []
            label_pls = []
            src_len_pls = []
            for i, device in enumerate(self._devices):
                with tf.device(device):
                    src_pls.append(
                        tf.placeholder(
                            dtype=tf.int32,
                            shape=[
                                None,
                                None],
                            name='src_pl_{}'.format(i)))
                    dst_pls.append(
                        tf.placeholder(
                            dtype=tf.int32,
                            shape=[
                                None,
                                None],
                            name='dst_pl_{}'.format(i)))
                    label_pls.append(
                        tf.placeholder(
                            dtype=tf.int32,
                            shape=[
                                None,
                                None],
                            name='label_pl_{}'.format(i)))
                    src_len_pls.append(
                        tf.placeholder(
                            dtype=tf.int32,
                            shape=[None],
                            name='src_len_pl_{}'.format(i)))
            self.src_pls = tuple(src_pls)
            self.dst_pls = tuple(dst_pls)
            self.label_pls = tuple(label_pls)
            self.src_len_pls = tuple(src_len_pls)

        self.encoder_scope = 'encoder'
        self.decoder_scope = 'decoder'
        self.decoder_label_scope = 'decoder_label'

    def prepare_training(self):
        with self.graph.as_default():
            # Optimizer
            self.global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                                               trainable=False, initializer=tf.zeros_initializer)

            self.learning_rate = tf.convert_to_tensor(
                self._config.train.learning_rate, dtype=tf.float32)
            if self._config.train.optimizer == 'adam':
                self._optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)
            elif self._config.train.optimizer == 'adam_decay':
                self.learning_rate *= learning_rate_decay(
                    self._config, self.global_step)
                self._optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
            elif self._config.train.optimizer == 'sgd':
                self._optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate)
            elif self._config.train.optimizer == 'mom':
                self._optimizer = tf.train.MomentumOptimizer(
                    self.learning_rate, momentum=0.9)

            # Uniform scaling initializer.
            self._initializer = init_ops.variance_scaling_initializer(
                scale=1.0, mode='fan_avg', distribution='uniform')

    def build_train_model(self, test=True, reuse=None):
        """Build model for training. """
        logging.info('Build train model.')
        self.prepare_training()

        with self.graph.as_default():
            acc_cop_list, acc_gen_list, acc_cop_edit_list, loss_list, gv_list = [], [], [], [], []
            cache = {}
            load = dict([(d, 0) for d in self._devices])
            for i, (X, Y, Z, X_len, device) in enumerate(
                    zip(self.src_pls, self.dst_pls, self.label_pls, self.src_len_pls, self._devices)):
                def daisy_chain_getter(getter, name, *args, **kwargs):
                    """Get a variable and cache in a daisy chain."""
                    device_var_key = (device, name)
                    if device_var_key in cache:
                        # if we have the variable on the correct device, return
                        # it.
                        return cache[device_var_key]
                    if name in cache:
                        # if we have it on a different device, copy it from the
                        # last device
                        v = tf.identity(cache[name])
                    else:
                        var = getter(name, *args, **kwargs)
                        v = tf.identity(
                            var._ref())  # pylint: disable=protected-access
                    # update the cache
                    cache[name] = v
                    cache[device_var_key] = v
                    return v

                def balanced_device_setter(op):
                    """Balance variables to all devices."""
                    if op.type in {'Variable', 'VariableV2', 'VarHandleOp'}:
                        # return self._sync_device
                        min_load = min(load.values())
                        min_load_devices = [
                            d for d in load if load[d] == min_load]
                        chosen_device = random.choice(min_load_devices)
                        load[chosen_device] += op.outputs[0].get_shape().num_elements()
                        return chosen_device
                    return device

                def identity_device_setter(op):
                    return device

                device_setter = balanced_device_setter

                with tf.variable_scope(tf.get_variable_scope(),
                                       initializer=self._initializer,
                                       custom_getter=daisy_chain_getter,
                                       reuse=reuse):
                    with tf.device(device_setter):
                        logging.info('Build model on %s.' % device)
                        encoder_output = self.encoder(
                            X, is_training=True, reuse=i > 0 or None)
                        decoder_output = self.decoder(
                            shift_right(Y),
                            encoder_output,
                            is_training=True,
                            reuse=i > 0 or None)

                        acc_cop, acc_gen, acc_cop_edit, loss = self.train_output_label(
                            decoder_output, Y, X, Z, reuse=i > 0 or None)

                        acc_cop_list.append(acc_cop)
                        acc_gen_list.append(acc_gen)
                        acc_cop_edit_list.append(acc_cop_edit)
                        loss_list.append(loss)

                        var_list = tf.trainable_variables()
                        if self._config.train.var_filter:
                            var_list = [
                                v for v in var_list if re.match(
                                    self._config.train.var_filter, v.name)]
                        gv_list.append(
                            self._optimizer.compute_gradients(
                                loss, var_list=var_list))

            self.accuracy_cop = tf.reduce_mean(acc_cop_list)
            self.accuracy_gen = tf.reduce_mean(acc_gen_list)
            self.accuracy_cop_edit = tf.reduce_mean(acc_cop_edit_list)
            self.loss = tf.reduce_mean(loss_list)

            # Clip gradients and then apply.
            grads_and_vars = average_gradients(gv_list)

            if self._config.train.grads_clip > 0:
                grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                                clip_norm=self._config.train.grads_clip)
                grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
            else:
                self.grads_norm = tf.global_norm(
                    [gv[0] for gv in grads_and_vars])

            self.train_op = self._optimizer.apply_gradients(
                grads_and_vars, global_step=self.global_step)

            # Summaries
            tf.summary.scalar('acc_cop', self.accuracy_cop)
            tf.summary.scalar('acc_gen', self.accuracy_gen)
            tf.summary.scalar('acc_cop_edit', self.accuracy_cop_edit)
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('learning_rate', self.learning_rate)
            tf.summary.scalar('grads_norm', self.grads_norm)
            self.summary_op = tf.summary.merge_all()

            self.saver = tf.train.Saver(var_list=tf.global_variables())

        # We may want to test the model during training.
        if test:
            self.build_test_model(reuse=True)

    def build_test_model(self, reuse=None):
        """Build model for inference."""
        logging.info('Build test model.')

        with self.graph.as_default(), tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

            cache = {}
            load = dict([(d, 0) for d in self._devices])

            prediction_list = []
            prediction_label_list = []
            loss_sum = 0
            loss_label_sum = 0
            for i, (X, Y, Z, X_lens, device) in enumerate(
                    zip(self.src_pls, self.dst_pls, self.label_pls, self.src_len_pls, self._devices)):
                def daisy_chain_getter(getter, name, *args, **kwargs):
                    """Get a variable and cache in a daisy chain."""
                    device_var_key = (device, name)
                    if device_var_key in cache:
                        # if we have the variable on the correct device, return
                        # it.
                        return cache[device_var_key]
                    if name in cache:
                        # if we have it on a different device, copy it from the
                        # last device
                        v = tf.identity(cache[name])
                    else:
                        var = getter(name, *args, **kwargs)
                        v = tf.identity(
                            var._ref())  # pylint: disable=protected-access
                    # update the cache
                    cache[name] = v
                    cache[device_var_key] = v
                    return v

                def balanced_device_setter(op):
                    """Balance variables to all devices."""
                    if op.type in {'Variable', 'VariableV2', 'VarHandleOp'}:
                        # return self._sync_device
                        min_load = min(load.values())
                        min_load_devices = [
                            d for d in load if load[d] == min_load]
                        chosen_device = random.choice(min_load_devices)
                        load[chosen_device] += op.outputs[0].get_shape().num_elements()
                        return chosen_device
                    return device

                device_setter = balanced_device_setter

                with tf.device(device):

                    logging.info('Build model on %s.' % device)
                    dec_input = shift_right(Y)
                    dec_label_input = shift_right(Z)

                    def true_fn():
                        enc_output = self.encoder(
                            X, is_training=False, reuse=i > 0 or None)

                        prediction, prediction_label = self.beam_search_label(
                            enc_output, X, X_lens, reuse=i > 0 or None)
                        dec_output = self.decoder(
                            dec_input, enc_output, is_training=False, reuse=True)

                        loss = self.test_loss_label(
                            dec_output, Y, Z, reuse=True)

                        return prediction, prediction_label, loss

                    def false_fn():

                        return tf.zeros([0, 0], dtype=tf.int32), tf.zeros(
                            [0, 0], dtype=tf.int32), 0.0


                    prediction, prediction_label, loss = tf.cond(
                        tf.greater(tf.shape(X)[0], 0), true_fn, false_fn)

                    loss_sum += loss
                    prediction_list.append(prediction)
                    prediction_label_list.append(prediction_label)

            max_length = tf.reduce_max([tf.shape(pred)[1]
                                        for pred in prediction_list])
            max_length_label = tf.reduce_max(
                [tf.shape(pred)[1] for pred in prediction_label_list])

            def pad_to_max_length(input, length):
                """Pad the input (with rank 2) with 3(</S>) to the given length in the second axis."""
                shape = tf.shape(input)
                padding = tf.ones(
                    [shape[0], length - shape[1]], dtype=tf.int32) * 3
                return tf.concat([input, padding], axis=1)

            # calculate the prediction of word sequences
            prediction_list = [
                pad_to_max_length(
                    pred, max_length) for pred in prediction_list]
            self.prediction = tf.concat(prediction_list, axis=0)

            # calculate the prediction of label sequences
            prediction_label_list = [
                pad_to_max_length(
                    pred, max_length_label) for pred in prediction_label_list]
            self.prediction_label = tf.concat(prediction_label_list, axis=0)

            self.loss_sum = loss_sum

            # Summaries
            tf.summary.scalar('loss_test', self.loss_sum)

            self.saver = tf.train.Saver(var_list=tf.global_variables())

    def encoder(self, encoder_input, is_training, reuse):
        """Encoder."""
        with tf.variable_scope(self.encoder_scope, reuse=reuse):
            return self.encoder_impl(encoder_input, is_training)

    def decoder(self, decoder_input, encoder_output, is_training, reuse):
        """Decoder"""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            return self.decoder_impl(decoder_input, encoder_output, is_training)

    def decoder_label(self, decoder_input, encoder_output, is_training, reuse):
        """Decoder"""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            return self.decoder_label_impl(
                decoder_input, encoder_output, is_training)

    def decoder_with_caching(
            self, decoder_input, decoder_cache, encoder_output, is_training, reuse):
        """Incremental Decoder"""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            return self.decoder_with_caching_impl(
                decoder_input, decoder_cache, encoder_output, is_training)

    def decoder_label_with_caching(
            self, decoder_input, decoder_cache, encoder_output, is_training, reuse):
        """Incremental Decoder"""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            return self.decoder_label_with_caching_impl(
                decoder_input, decoder_cache, encoder_output, is_training)

    def beam_search(self, encoder_output, encoder_input, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self._config.test.beam_size, tf.shape(encoder_output)[
            0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, *, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        encoder_output = tf.reshape(encoder_output, [
                                    batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1),
                             dtype=tf.float32)  # [beam_size]
        # [batch_size * beam_size]
        scores = tf.tile(scores, multiples=[batch_size])
        bias = tf.zeros_like(scores, dtype=tf.bool)
        cache = tf.zeros([batch_size * beam_size, 0,
                          self._config.num_blocks, self._config.hidden_units])

        def step(i, bias, preds, scores, cache):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            decoder_output, cache = self.decoder_with_caching(
                preds, cache, encoder_output, is_training=False, reuse=reuse)
            # there remains to rewrite

            last_preds, last_k_preds, last_k_scores = self.test_output(
                decoder_output, encoder_input, reuse=reuse)

            last_k_preds = get_bias_preds(last_k_preds, bias)
            last_k_scores = get_bias_scores(last_k_scores, bias)

            # Update scores.
            # [batch_size * beam_size, beam_size]
            scores = scores[:, None] + last_k_scores
            # [batch_size, beam_size * beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])

            # Pruning.
            scores, k_indices = tf.nn.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, shape=[-1])  # [batch_size * beam_size]
            base_indices = tf.reshape(
                tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            # [batch_size * beam_size]
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])

            # Update predictions.
            last_k_preds = tf.gather(tf.reshape(
                last_k_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(
                preds, indices=tf.cast(
                    k_indices / beam_size, tf.int64))
            cache = tf.gather(
                cache, indices=tf.cast(
                    k_indices / beam_size, tf.int64))
            # [batch_size * beam_size, i]
            preds = tf.concat((preds, last_k_preds[:, None]), axis=1)

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            return i, bias, preds, scores, cache

        def not_finished(i, bias, preds, scores, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] +
                                   50, self._config.test.max_target_length])
                )
            )

        i, bias, preds, scores, cache = tf.while_loop(cond=not_finished,
                                                      body=step,
                                                      loop_vars=[
                                                          0, bias, preds, scores, cache],
                                                      shape_invariants=[
                                                          tf.TensorShape([]),
                                                          tf.TensorShape(
                                                              [None]),
                                                          tf.TensorShape(
                                                              [None, None]),
                                                          tf.TensorShape(
                                                              [None]),
                                                          tf.TensorShape([None, None, None, None])],
                                                      back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        # [batch_size, beam_size, max_length]
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])
        lengths = tf.reduce_sum(
            tf.to_float(
                tf.not_equal(
                    preds,
                    3)),
            axis=-
            1)   # [batch_size, beam_size]
        lp = tf.pow((5 + lengths) / (5 + 1),
                    self._config.test.lp_alpha)   # Length penalty
        scores /= lp   # following GNMT
        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))   # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag
        return final_preds

    def beam_search_label(self, encoder_output, encoder_input, encoder_input_lens, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self._config.test.beam_size, tf.shape(encoder_output)[0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, *, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        encoder_output = tf.reshape(encoder_output, [
                                    batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1),
                             dtype=tf.float32)  # [beam_size]
        # [batch_size * beam_size]
        scores = tf.tile(scores, multiples=[batch_size])
        bias = tf.zeros_like(scores, dtype=tf.bool)
        cache = tf.zeros([batch_size * beam_size, 0,
                          self._config.num_blocks, self._config.hidden_units])

        # for label prediction
        preds_label = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores_label = tf.constant(
            [0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores_label = tf.tile(
            scores_label,
            multiples=[batch_size])  # [batch_size * beam_size]
        bias_label = tf.zeros_like(scores_label, dtype=tf.bool)
        cache_label = tf.zeros(
            [batch_size * beam_size, 0, self._config.num_blocks, self._config.hidden_units])

        def step(i, bias, preds, scores, cache, bias_label,
                 preds_label, scores_label, cache_label):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            decoder_output, cache = self.decoder_with_caching(
                preds, cache, encoder_output, is_training=False, reuse=reuse)

            last_preds, last_k_preds, last_k_scores = self.test_output(
                decoder_output, encoder_input, reuse=reuse)

            last_k_preds = get_bias_preds(last_k_preds, bias)
            last_k_scores = get_bias_scores(last_k_scores, bias)

            # Update scores.
            # [batch_size * beam_size, beam_size]
            scores = scores[:, None] + last_k_scores
            # [batch_size, beam_size * beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])

            # Pruning.
            scores, k_indices = tf.nn.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, shape=[-1])  # [batch_size * beam_size]
            base_indices = tf.reshape(
                tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            # [batch_size * beam_size]
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])

            # Update predictions.
            last_k_preds = tf.gather(tf.reshape(
                last_k_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(
                preds, indices=tf.cast(
                    k_indices / beam_size, tf.int64))
            cache = tf.gather(
                cache, indices=tf.cast(
                    k_indices / beam_size, tf.int64))
            # [batch_size * beam_size, i]
            preds = tf.concat((preds, last_k_preds[:, None]), axis=1)

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            # caculate parameter related to label
            last_preds_label, last_k_preds_label, last_k_scores_label = self.test_output_label(decoder_output,
                                                                                               encoder_input,
                                                                                               reuse=reuse)

            last_k_preds_label = get_bias_preds(last_k_preds_label, bias_label)
            last_k_scores_label = get_bias_scores(
                last_k_scores_label, bias_label)

            # Update scores.
            # [batch_size * beam_size, beam_size]
            scores_label = scores_label[:, None] + last_k_scores_label
            # [batch_size, beam_size * beam_size]
            scores_label = tf.reshape(
                scores_label, shape=[
                    batch_size, beam_size ** 2])

            # Pruning.
            scores_label, k_indices = tf.nn.top_k(scores_label, k=beam_size)
            # [batch_size * beam_size]
            scores_label = tf.reshape(scores_label, shape=[-1])
            base_indices = tf.reshape(
                tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            # [batch_size * beam_size]
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])

            # Update predictions.
            last_k_preds_label = tf.gather(tf.reshape(
                last_k_preds_label, shape=[-1]), indices=k_indices)
            preds_label = tf.gather(
                preds_label, indices=tf.cast(
                    k_indices / beam_size, tf.int64))
            cache_label = tf.gather(
                cache_label, indices=tf.cast(
                    k_indices / beam_size, tf.int64))
            # [batch_size * beam_size, i]
            preds_label = tf.concat(
                (preds_label, last_k_preds_label[:, None]), axis=1)

            # Whether sequences finished.
            bias_label = tf.equal(preds_label[:, -1], 3)  # </S>?

            return i, bias, preds, scores, cache, bias_label, preds_label, scores_label, cache_label

        def not_finished(i, bias, preds, scores, cache, bias_label,
                         preds_label, scores_label, cache_label):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.shape(encoder_output)[1]
                )
            )  # not all sequences have decoded the '</S>', and not exceed the max_length

        i, bias, preds, scores, cache, bias_label, preds_label, scores_label, cache_label \
            = tf.while_loop(cond=not_finished,
                          body=step,
                          loop_vars=[
                              0, bias, preds, scores, cache, bias_label, preds_label, scores_label, cache_label],
                            shape_invariants=[
                              tf.TensorShape(
                                  []),
                              tf.TensorShape(
                                  [None]),
                              tf.TensorShape(
                                  [None, None]),
                              tf.TensorShape(
                                  [None]),
                              tf.TensorShape(
                                  [None, None, None, None]),
                              tf.TensorShape(
                                  [None]),
                              tf.TensorShape(
                                  [None, None]),
                              tf.TensorShape(
                                  [None]),
                              tf.TensorShape([None, None, None, None])],
                            back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        # [batch_size, beam_size, max_length]
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])
        lengths = tf.reduce_sum(
            tf.to_float(
                tf.not_equal(
                    preds,
                    3)),
            axis=-
            1)   # [batch_size, beam_size]
        lp = tf.pow((5 + lengths) / (5 + 1),
                    self._config.test.lp_alpha)   # Length penalty
        scores /= lp                                                     # following GNMT
        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))   # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag

        # caculate parameter related to label
        scores_label = tf.reshape(scores_label, shape=[batch_size, beam_size])
        # [batch_size, beam_size, max_length]
        preds_label = tf.reshape(preds_label, shape=[batch_size, beam_size, -1])
        lengths_label = tf.reduce_sum(
            tf.to_float(
                tf.not_equal(
                    preds_label,
                    3)),
            axis=-
            1)  # [batch_size, beam_size]
        lp = tf.pow((5 + lengths_label) / (5 + 1),
                    self._config.test.lp_alpha)  # Length penalty
        scores_label /= lp  # following GNMT
        max_indices_label = tf.to_int32(
            tf.argmax(scores_label, axis=-1))  # [batch_size]
        max_indices_label += tf.range(batch_size) * beam_size
        preds_label = tf.reshape(
            preds_label, shape=[
                batch_size * beam_size, -1])

        final_preds_label = tf.gather(preds_label, indices=max_indices)
        final_preds_label = final_preds_label[:, 1:]  # remove <S> flag

        return final_preds, final_preds_label

    def test_output(self, decoder_output, encoder_input, reuse):
        """During test, we only need the last prediction at each time."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            def mask(input_pl, logits):
                input_pl = input_pl[:, :]
                batch_size = tf.cast(tf.shape(input_pl), dtype=tf.int32)[0]
                input_pl = tf.concat([input_pl, tf.reshape(tf.tile(
                    [tf.constant(3, dtype=tf.int32)], [batch_size]), [batch_size, 1])], 1)
                input_pl = tf.concat([input_pl, tf.reshape(tf.tile(
                    [tf.constant(1, dtype=tf.int32)], [batch_size]), [batch_size, 1])], 1)
                input_pl = tf.concat([input_pl, tf.reshape(tf.tile(
                    [tf.constant(2, dtype=tf.int32)], [batch_size]), [batch_size, 1])], 1)
                input_pl = tf.concat([input_pl, tf.reshape(tf.tile(
                    [tf.constant(0, dtype=tf.int32)], [batch_size]), [batch_size, 1])], 1)
                input_pl = tf.concat([input_pl, tf.reshape(tf.tile([tf.constant(
                    4, dtype=tf.int32)], [batch_size]), [batch_size, 1])], 1)

                seq_length = tf.cast(tf.shape(input_pl), dtype=tf.int32)[1]
                vocab_size = tf.constant(
                    self._config.dst_vocab_size, dtype=tf.int64)
                batch_index = tf.reshape(tf.tile(tf.range(batch_size), [
                                         seq_length]), [-1, batch_size])

                mask_index = tf.reshape(
                    tf.stack([batch_index, tf.transpose(input_pl)], 2), [-1, 2])
                mask_index = tf.cast(mask_index, dtype=tf.int64)
                mask_value = tf.tile(
                    tf.tile([tf.constant(1, dtype=tf.float32)], [batch_size]), [seq_length])
                mask = tf.SparseTensor(values=mask_value, indices=mask_index,
                                       dense_shape=[tf.cast(batch_size, dtype=tf.int64), vocab_size])
                mul_mask = tf.sparse_tensor_to_dense(
                    tf.sparse_reorder(mask), validate_indices=False)
                comparison = tf.equal(mul_mask, tf.constant(0.0))
                batch_size = tf.cast(batch_size, dtype=tf.int64)
                mask_inf = tf.ones_like(mul_mask, dtype=tf.float32) * (-1.0)
                add_mask = tf.where(
                    comparison, mask_inf, tf.ones_like(
                        mul_mask, dtype=tf.float32))
                mul_mask = tf.tile(
                    mul_mask, multiples=[
                        self._config.test.beam_size, 1])

                return tf.multiply(mul_mask, logits)
            last_logits_tmp = dense(decoder_output[:, -1], self._config.dst_vocab_size, use_bias=False,
                                    name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                                    reuse=True if self._config.tie_embedding_and_softmax else None)
            last_logits = mask(encoder_input, last_logits_tmp)

            last_preds = tf.to_int32(tf.argmax(last_logits, axis=-1))

            z = tf.nn.log_softmax(last_logits)
            last_k_scores, last_k_preds = tf.nn.top_k(
                z, k=self._config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_output_label(self, decoder_output, encoder_input, reuse):
        """During test, we only need the last prediction at each time."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            last_label_logits = dense(decoder_output[:, -1], self._config.lbl_vocab_size, use_bias=False,
                                      name="lbl_embedding",
                                      reuse=reuse)

            last_preds = tf.to_int32(tf.argmax(last_label_logits, axis=-1))

            z = tf.nn.log_softmax(last_label_logits)
            last_k_scores, last_k_preds = tf.nn.top_k(
                z, k=self._config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_loss(self, decoder_output, Y, reuse):
        """This function help users to compute PPL during test."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            logits = dense(decoder_output, self._config.dst_vocab_size, use_bias=False,
                           name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                           reuse=True if self._config.tie_embedding_and_softmax else None)
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=self._config.dst_vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=labels)
            loss_sum = tf.reduce_sum(loss * mask)
        return loss_sum

    def test_loss_label(self, decoder_output, Y, Z, reuse):
        """This function help users to compute PPL during test."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            # calculate the loss of generation
            logits_gen = dense(decoder_output, self._config.dst_vocab_size, use_bias=False,
                               name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                               reuse=True if self._config.tie_embedding_and_softmax else None)
            mask_gen = tf.to_float(tf.not_equal(Y, 0))
            labels_gen = tf.one_hot(Y, depth=self._config.dst_vocab_size)
            loss_gen = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits_gen, labels=labels_gen)
            loss_sum_gen = tf.reduce_sum(
                loss_gen * mask_gen) / (tf.reduce_sum(mask_gen))

            # calculate the loss of label
            logits_cop = dense(decoder_output, self._config.lbl_vocab_size, use_bias=False,
                               name="lbl_embedding",
                               reuse=reuse)
            mask_cop = tf.to_float(tf.not_equal(Z, 0))
            labels_cop = tf.one_hot(Z, depth=self._config.lbl_vocab_size)
            loss_cop = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits_cop, labels=labels_cop)
            loss_sum_cop = tf.reduce_sum(
                loss_cop * mask_cop) / (tf.reduce_sum(mask_cop))

            if self._config.label_loss_ratio:
                # calculate the weighted average loss
                mean_loss_test = tf.add(
                    self._config.label_loss_ratio * loss_sum_cop,
                    (1.0 - self._config.label_loss_ratio) * loss_sum_gen)
            else:
                mean_loss_test = loss_sum_gen
        return mean_loss_test

    def train_output(self, decoder_output, Y, X, reuse):
        """Calculate loss and accuracy."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            logits_gen = dense(decoder_output, self._config.dst_vocab_size, use_bias=False,
                               name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                               reuse=True if self._config.tie_embedding_and_softmax else None)  # 2D to 3D

            preds_gen = tf.to_int32(tf.argmax(logits_gen, axis=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc_gen = tf.reduce_sum(
                tf.to_float(
                    tf.equal(
                        preds_gen,
                        Y)) * mask) / tf.reduce_sum(mask)

            # Smoothed loss
            loss_gen = common_layers.smoothing_cross_entropy(logits=logits_gen, labels=Y,
                                                             vocab_size=self._config.dst_vocab_size,
                                                             confidence=1 - self._config.train.label_smoothing)
            mean_loss_gen = tf.reduce_sum(
                loss_gen * mask) / (tf.reduce_sum(mask))

        return acc_gen, mean_loss_gen

    def train_output_label(self, decoder_output, Y, X, Z, reuse):
        """Calculate loss and accuracy."""
        with tf.variable_scope(self.decoder_scope, reuse=reuse):
            logits_gen = dense(decoder_output, self._config.dst_vocab_size, use_bias=True,
                               name="dst_embedding" if self._config.tie_embedding_and_softmax else "softmax",
                               reuse=True if self._config.tie_embedding_and_softmax else None)  # 2D to 3D

            logits_cop = dense(decoder_output, self._config.lbl_vocab_size, use_bias=True,
                               name="lbl_embedding",
                               reuse=reuse)
            p_copy = tf.sigmoid(logits_cop)

            loss_copy = common_layers.smoothing_cross_entropy(logits=logits_cop, labels=Z,
                                                              vocab_size=self._config.lbl_vocab_size,
                                                              confidence=1 - self._config.train.label_smoothing)

            mask = tf.to_float(tf.not_equal(Z, 0))
            preds_cop = tf.to_int32(tf.argmax(logits_cop, axis=-1))
            acc_cop = tf.reduce_sum(
                tf.to_float(
                    tf.equal(
                        preds_cop,
                        Z)) * mask) / tf.reduce_sum(mask)
            mean_loss_cop = tf.reduce_sum(
                loss_copy * mask) / (tf.reduce_sum(mask))

            # calculate accuracy of edit label
            Z_edit = tf.to_int32(tf.equal(Z, 2))
            preds_cop_edit = tf.to_int32(tf.equal(preds_cop, 2))
            mask_edit = tf.to_float(tf.not_equal(Z_edit, 0))
            acc_cop_edit = tf.reduce_sum(tf.to_float(tf.equal(preds_cop_edit, Z_edit)) * mask_edit) / tf.reduce_sum(
                mask_edit)

            preds_gen = tf.to_int32(tf.argmax(logits_gen, axis=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc_gen = tf.reduce_sum(
                tf.to_float(
                    tf.equal(
                        preds_gen,
                        Y)) * mask) / tf.reduce_sum(mask)

            # Smoothed loss
            loss_gen = common_layers.smoothing_cross_entropy(logits=logits_gen, labels=Y,
                                                             vocab_size=self._config.dst_vocab_size,
                                                             confidence=1 - self._config.train.label_smoothing)
            if self._config.del_penalty_coef:
                penalty_mask = tf.to_float(tf.equal(
                    preds_cop, 2)) * self._config.del_penalty_coef + tf.to_float(tf.not_equal(preds_cop, 2))
                loss_gen = loss_gen * penalty_mask

            mean_loss_gen = tf.reduce_sum(
                loss_gen * mask) / (tf.reduce_sum(mask))

            if self._config.label_loss_ratio:
                mean_loss = tf.add(
                    self._config.label_loss_ratio * mean_loss_cop,
                    (1.0 - self._config.label_loss_ratio) * mean_loss_gen)
            else:
                mean_loss = mean_loss_gen

        return acc_cop, acc_gen, acc_cop_edit, mean_loss

    def encoder_impl(self, encoder_input, is_training):
        """
        This is an interface leave to be implemented by sub classes.
        Args:
            encoder_input: A tensor with shape [batch_size, src_length]

        Returns: A Tensor with shape [batch_size, src_length, num_hidden]

        """
        raise NotImplementedError()

    def decoder_impl(self, decoder_input, encoder_output, is_training):
        """
        This is an interface leave to be implemented by sub classes.
        Args:
            decoder_input: A Tensor with shape [batch_size, dst_length]
            encoder_output: A Tensor with shape [batch_size, src_length, num_hidden]

        Returns: A Tensor with shape [batch_size, dst_length, num_hidden]

        """
        raise NotImplementedError()

    def decoder_with_caching_impl(
            self, decoder_input, decoder_cache, encoder_output, is_training):
        """
        This is an interface leave to be implemented by sub classes.
        Args:
            decoder_input: A Tensor with shape [batch_size, dst_length]
            decoder_cache: A Tensor with shape [batch_size, *, *, num_hidden]
            encoder_output: A Tensor with shape [batch_size, src_length, num_hidden]

        Returns: A Tensor with shape [batch_size, dst_length, num_hidden]

        """
        raise NotImplementedError()

    def decoder_label_with_caching_impl(
            self, decoder_input, decoder_cache, encoder_output, is_training):
        """
        This is an interface leave to be implemented by sub classes.
        Args:
            decoder_input: A Tensor with shape [batch_size, dst_length]
            decoder_cache: A Tensor with shape [batch_size, *, *, num_hidden]
            encoder_output: A Tensor with shape [batch_size, src_length, num_hidden]

        Returns: A Tensor with shape [batch_size, dst_length, num_hidden]

        """
        raise NotImplementedError()

    def decoder_label_impl(self, decoder_input, encoder_output, is_training):
        """
        This is an interface leave to be implemented by sub classes.
        Args:
            decoder_input: A Tensor with shape [batch_size, dst_length]
            encoder_output: A Tensor with shape [batch_size, src_length, num_hidden]

        Returns: A Tensor with shape [batch_size, dst_length, num_hidden]

        """
        raise NotImplementedError()
