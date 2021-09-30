# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:23:12 2018

@author: cm
"""

import os
import sys

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
from sentence_segmentation_dl.utils import load_vocabulary


class Hyperparamters:
    # Train parameters
    num_train_epochs = 20
    print_step = 10
    batch_size = 16  # 64
    summary_step = 10
    num_saved_per_epoch = 3
    max_to_keep = 100
    logdir = 'logdir/model_02'
    file_save_model = 'model/model_02'
    file_load_model = 'model/V1.1'

    # Train/Test data
    data_dir = os.path.join(pwd, 'data')
    train_data = '20201203/data_train.csv'
    test_data = '20201203/data_test.csv'

    # Load vocabulcary dict
    dict_id2label, dict_label2id = load_vocabulary(os.path.join(pwd, 'data', 'vocabulary_label.txt'))
    label_vocabulary = list(dict_id2label.values())

    # Optimization parameters
    warmup_proportion = 0.1
    use_tpu = None
    do_lower_case = True
    learning_rate = 5e-5

    # LSTM parameters
    lstm_hidden_size = 128
    keep_prob = 0.5

    # Sequence and Label
    sequence_length = 128 - 2
    num_labels = len(list(dict_id2label))

    # ALBERT
    model = 'albert_small_zh_google'
    bert_path = os.path.join(pwd, model)
    vocab_file = os.path.join(pwd, model, 'vocab_chinese.txt')
    init_checkpoint = os.path.join(pwd, model, 'albert_model.ckpt')
    saved_model_path = os.path.join(pwd, 'model')


if __name__ == '__main__':
    hp = Hyperparamters()
    print(hp.dict_id2label)
