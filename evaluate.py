from __future__ import print_function

import codecs
import subprocess
import os
import time
from argparse import ArgumentParser
from tempfile import mkstemp

import numpy as np
import yaml
import logging

from models import *
from utils import DataReader, AttrDict, expand_feed_dict, print_num_of_total_parameters, available_variables

import tensorflow as tf


class Evaluator(object):
    """
    Evaluate the model.
    """
    def __init__(self):
        pass

    def init_from_config(self, config):

        logger = logging.getLogger('')

        self.model = eval(config.model)(config, config.test.num_gpus)
        self.model.build_test_model()

        # Print the number of total parameters
        print_num_of_total_parameters()

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.Session(config=sess_config, graph=self.model.graph)
        # Restore model.
        self.model.saver.restore(self.sess, tf.train.latest_checkpoint(config.model_dir))

        self.data_reader = DataReader(config)

    def init_from_existed(self, model, sess, data_reader):
        assert model.graph == sess.graph
        self.sess = sess
        self.model = model
        self.data_reader = data_reader

    def beam_search(self, X):
        return self.sess.run(self.model.prediction, feed_dict=expand_feed_dict({self.model.src_pls: X}))

    def beam_search_label(self, X, Y, Z, X_lens):
        return self.sess.run([self.model.prediction, self.model.prediction_label], feed_dict=expand_feed_dict({self.model.src_pls: X, self.model.dst_pls: Y, self.model.label_pls: Z, self.model.src_len_pls: X_lens}))

    def loss(self, X, Y):
        return self.sess.run(self.model.loss_sum, feed_dict=expand_feed_dict({self.model.src_pls: X, self.model.dst_pls: Y}))

    def loss_label(self, X, Y, Z):
        return self.sess.run(self.model.loss_sum, feed_dict=expand_feed_dict({self.model.src_pls: X, self.model.dst_pls: Y, self.model.label_pls: Z}))

    def translate(self, src_path, dst_path, lbl_path, output_path, output_label_path, batch_size):
        logging.info('Translate %s.' % src_path)
        _, tmp = mkstemp()
        fd = codecs.open(tmp, 'w', 'utf8')

        _, tmp_label = mkstemp()
        fd_label = codecs.open(tmp_label, 'w', 'utf8')

        count = 0
        token_count = 0
        start = time.time()
        for X, ref, label, src_lens in self.data_reader.get_test_batches_with_target_with_label(src_path, dst_path, lbl_path, batch_size):
            Y, Z = self.beam_search_label(X, ref, label, src_lens)
            sents = self.data_reader.indices_to_words(Y, src_lens)
            assert len(X) == len(sents)
            for sent in sents:
                print(sent, file=fd)
            count += len(X)
            token_count += np.sum(np.not_equal(Y, 3))  # 3: </s>
            time_span = time.time() - start
            logging.info('{0} sentences ({1} tokens) processed in {2:.2f} minutes (speed: {3:.4f} sec/token).'.
                         format(count, token_count, time_span / 60, time_span / token_count))

            # Save the prediction of label
            sents_label = self.data_reader.indices_to_words(Z, src_lens, o='lbl')
            assert len(X) == len(sents_label)
            for sent in sents_label:
                print(sent, file=fd_label)

        fd.close()

        # Remove BPE flag, if have.
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp, output_path))
        os.remove(tmp)
        logging.info('The result file was saved in %s.' % output_path)

        fd_label.close()
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' %s > %s" % (tmp_label, output_label_path))
        os.remove(tmp_label)
        logging.info('The label file was saved in %s.' % output_label_path)

    def ppl(self, src_path, dst_path, batch_size):
        logging.info('Calculate PPL for %s and %s.' % (src_path, dst_path))
        token_count = 0
        loss_sum = 0
        for batch in self.data_reader.get_test_batches_with_target(src_path, dst_path, batch_size):
            X, Y = batch
            loss_sum += self.loss(X, Y)
            token_count += np.sum(np.greater(Y, 0))
        # Compute PPL
        ppl = np.exp(loss_sum / token_count)
        logging.info('PPL: %.4f' % ppl)
        return ppl

    def fscore(self, lbl_path, output_label_path):
        logging.info('Calculate P/R/F for %s and %s.' % (lbl_path, output_label_path))
        ref_file = codecs.open(lbl_path, 'r', 'utf8')
        pred_file = codecs.open(output_label_path, 'r', 'utf8')

        tp, fp, fn = 1, 1, 1
        err = 0
        # assert len(target) == len(prediction)
        line = 0
        for ref, pred in zip(ref_file, pred_file):
            line += 1
            if len(ref) != len(pred):
                # print(line)
                err += 1
                continue
            for x, y in zip(ref, pred):
                if x == y and x == 'E':
                    tp += 1
                elif y == 'E':
                    fp += 1
                elif x == 'E':
                    fn += 1
                else:
                    pass
        print('tp:{}, fp:{}, fn:{}, err:{}'.format(tp, fp, fn, err))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = (2 * precision * recall / (precision + recall))

        ref_file.close()
        pred_file.close()

        logging.info('precision: %.4f' % precision)
        logging.info('recall: %.4f' % recall)
        logging.info('fscore: %.4f' % fscore)
        return precision, recall, fscore

    def evaluate(self, batch_size, **kargs):
        """Evaluate the model on dev set."""
        src_path = kargs['src_path']
        dst_path = kargs['ref_path']
        lbl_path = kargs['label_path']
        output_path = kargs['output_path']
        output_label_path = kargs['output_label_path']
        cmd = kargs['cmd'] if 'cmd' in kargs else\
            "perl multi-bleu.perl {ref} < {output} 2>/dev/null | awk '{{print($3)}}' | awk -F, '{{print $1}}'"
        self.translate(src_path, dst_path, lbl_path, output_path, output_label_path, batch_size)

        if 'dst_path' in kargs:
            self.ppl(src_path, kargs['dst_path'], batch_size)

        # calculate the fscore of label result
        if 'label_path' in kargs:
            precision, recall, f_score = self.fscore(lbl_path, output_label_path)
            return float(f_score)

        return None


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    logging.basicConfig(level=logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    evaluator = Evaluator()
    evaluator.init_from_config(config)
    for attr in config.test:
        if attr.startswith('set'):
            evaluator.evaluate(config.test.batch_size, **config.test[attr])
    logging.info("Done")
