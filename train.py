import os
import time
from argparse import ArgumentParser
import yaml

from evaluate import Evaluator
from models import *
from utils import DataReader, AttrDict, available_variables, expand_feed_dict, print_num_of_total_parameters
import datetime
import logging
import tensorflow as tf

def train(config):
    logger = logging.getLogger('')

    """Train a model with a config file."""
    data_reader = DataReader(config=config)
    model = eval(config.model)(config=config, num_gpus=config.train.num_gpus)
    model.build_train_model(test=config.train.eval_on_dev)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    summary_writer = tf.summary.FileWriter(config.model_dir, graph=model.graph)

    with tf.Session(config=sess_config, graph=model.graph) as sess:
        # Initialize all variables.
        sess.run(tf.global_variables_initializer())

        # Print the number of total parameters
        print_num_of_total_parameters()

        # Reload variables in disk.
        if tf.train.latest_checkpoint(config.model_dir):
            available_vars = available_variables(config.model_dir)
            if available_vars:
                saver = tf.train.Saver(var_list=available_vars)
                saver.restore(sess, tf.train.latest_checkpoint(config.model_dir))
                for v in available_vars:
                    logger.info('Reload {} from disk.'.format(v.name))
            else:
                logger.info('Nothing to be reload from disk.')
        else:
            logger.info('Nothing to be reload from disk.')

        evaluator = Evaluator()
        evaluator.init_from_existed(model, sess, data_reader)

        global dev_fscore, toleration
        dev_fscore = evaluator.evaluate(**config.dev) if config.train.eval_on_dev else 0
        toleration = config.train.toleration

        def train_one_step(batch):
            feed_dict = expand_feed_dict({model.src_pls: batch[0], model.dst_pls: batch[1], model.label_pls: batch[2], model.src_len_pls: batch[3]})

            step, lr, loss, _ = sess.run(
                [model.global_step, model.learning_rate,
                 model.loss, model.train_op],
                feed_dict=feed_dict)
            if step % config.train.summary_freq == 0:
                summary = sess.run(model.summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=step)
            return step, lr, loss

        def maybe_save_model():
            global dev_fscore, toleration
            new_dev_fscore = evaluator.evaluate(**config.dev) if config.train.eval_on_dev else dev_fscore + 1
            if new_dev_fscore >= dev_fscore:
                mp = config.model_dir + '/model_step_{}'.format(step)
                model.saver.save(sess, mp)
                logger.info('Save model in %s.' % mp)
                toleration = config.train.toleration
                dev_fscore = new_dev_fscore
            else:
                toleration -= 1

        step = 0
        logger.info("Begin training.")
        train_start_time = datetime.datetime.now()

        for epoch in range(1, config.train.num_epochs+1):
            for batch in data_reader.get_training_batches_with_buckets():

                # Train normal instances.
                start_time = time.time()
                step, lr, loss = train_one_step(batch)
                logger.info(
                    'epoch: {0}\tstep: {1}\tlr: {2:.6f}\tloss: {3:.4f}\ttime: {4:.4f}'.
                    format(epoch, step, lr, loss, time.time() - start_time))

                # Save model
                if config.train.save_freq > 0 and step % config.train.save_freq == 0:
                    maybe_save_model()

                if config.train.num_steps and step >= config.train.num_steps:
                    break

            # Save model per epoch if config.train.save_freq is less or equal than zero
            if config.train.save_freq <= 0:
                maybe_save_model()

            # Early stop
            if toleration <= 0:
                break
        train_end_time = datetime.datetime.now()
        interval = (train_end_time - train_start_time).seconds
        final_time = interval / 60.0 / 60.0  # convert to hours
        logger.info("Total training time: {0}".format(final_time))
        logger.info("Finish training.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    parser.add_argument('-t', '--test', dest='None')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    logging.basicConfig(filename=config.model_dir + '/train.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger('').addHandler(console)
    # Train
    train(config)
