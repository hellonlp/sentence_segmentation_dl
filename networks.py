# -*- coding: utf-8 -*-
"""
Created on Thu May 30 20:44:42 2019

@author: cm
"""

import os
import tensorflow as tf
from sentence_segmentation_dl import modeling
from sentence_segmentation_dl import optimization
from sentence_segmentation_dl.modules import cell_bilstm, cell_dense
from sentence_segmentation_dl.utils import time_now_string
from sentence_segmentation_dl.hyperparameters import Hyperparamters as hp
from sentence_segmentation_dl.ner_utils import NERProcessor

processor = NERProcessor()
bert_config_file = os.path.join(hp.bert_path, 'albert_config.json')
bert_config = modeling.AlbertConfig.from_json_file(bert_config_file)


class NetworkAlbertNER(object):
    def __init__(self, is_training):

        # Placeholder       
        self.input_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_ids')
        self.input_masks = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='input_masks')
        self.segment_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='segment_ids')
        self.label_ids = tf.placeholder(tf.int32, shape=[None, hp.sequence_length], name='label_ids')

        # Training or not
        self.is_training = is_training
        # Load BERT model
        self.model = modeling.AlbertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_masks,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False)

        # Get the feature vector by BERT
        output_layer_init = self.model.get_sequence_output()

        # Cell cnn
        # output_layer = cell_bilstm(output_layer_init,self.is_training)
        output_layer = cell_bilstm(output_layer_init, hp.lstm_hidden_size, is_training, name_scope="cell_bilstm")
        print('output_layer:', output_layer)  # (?, 200, 128)

        # Logit       
        self.logits = cell_dense(output_layer, hp.num_labels)
        print('logits:', self.logits)  # (?, 200, 2)
        # self.outputs =

        # Input length
        self.input_ids_sequence_length = tf.count_nonzero(output_layer_init,
                                                          axis=2,
                                                          dtype=tf.int32)
        
        self.input_ids_length = tf.count_nonzero(self.input_ids_sequence_length,
                                                 axis=1, dtype=tf.int32)

        # print('input_ids_sequence_length:',self.input_ids_sequence_length) #(?, 200)
        print('input_ids_length:', self.input_ids_length)  # (?,)

        print('self.logits:', self.logits)  # (?, 200, 2)
        print('self.label_ids:', self.label_ids)  # (?, 200)
        print('self.input_ids_length:', self.input_ids_length)  # (?,)

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits,
                                                                                        self.label_ids,
                                                                                        self.input_ids_length)

        with tf.variable_scope("loss"):
            # Checkpoint
            ckpt = tf.train.get_checkpoint_state(hp.saved_model_path)
            checkpoint_suffix = ".index"
            if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
                print('=' * 10, 'Restoring model from checkpoint!', '=' * 10)
                print("%s - Restoring model from checkpoint ~%s" % (time_now_string(),
                                                                    ckpt.model_checkpoint_path))
            else:
                print('=' * 10, 'First time load BERT model!', '=' * 10)
                tvars = tf.trainable_variables()
                if hp.init_checkpoint:
                    (assignment_map, initialized_variable_names) = \
                        modeling.get_assignment_map_from_checkpoint(tvars,
                                                                    hp.init_checkpoint)
                    tf.train.init_from_checkpoint(hp.init_checkpoint, assignment_map)

            if self.is_training:
                # Global_step
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.loss = tf.reduce_mean(-self.log_likelihood)

                # Optimizer BERT
                train_examples = processor.get_train_examples(hp.data_dir)
                num_train_steps = int(
                    len(train_examples) / hp.batch_size * hp.num_train_epochs)
                num_warmup_steps = int(num_train_steps * hp.warmup_proportion)
                print('num_train_steps', num_train_steps)
                self.optimizer = optimization.create_optimizer(self.loss,
                                                               hp.learning_rate,
                                                               num_train_steps,
                                                               num_warmup_steps,
                                                               hp.use_tpu,
                                                               Global_step=self.global_step)

                # Summary for tensorboard                 
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()


if __name__ == '__main__':
    # Load model
    albert = NetworkAlbertNER(is_training=True)
