# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:43:39 2018

@author: cm
"""

import time
import numpy as np
import pandas as pd


def time_now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def load_txt(file):
    with  open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
        print("Load data from file (%s) finished !" % file)
    return lines


def save_txt(file, lines):
    lines = [l + '\n' for l in lines]
    with  open(file, 'w+', encoding='utf-8') as fp:
        fp.writelines(lines)
    return "Write data to txt finished !"


def load_csv(file, encoding='utf-8', header=0):
    return pd.read_csv(file,
                       encoding=encoding,
                       header=header,
                       error_bad_lines=False)


def save_csv(dataframe, file, header=True, index=None, encoding="utf-8-sig"):
    return dataframe.to_csv(file,
                            mode='w+',
                            header=header,
                            index=index,
                            encoding=encoding)


def save_excel(dataframe, file, header=True, sheetname='Sheet1'):
    return dataframe.to_excel(file,
                              header=header,
                              sheet_name=sheetname)


def load_excel(file, header=0):
    return pd.read_excel(file,
                         header=header)
                         

def load_vocabulary(file_vocabulary_label):
    """
    Load vocabulary to dict
    """
    vocabulary = load_txt(file_vocabulary_label)
    dict_id2label, dict_label2id = {}, {}
    for i, l in enumerate(vocabulary):
        dict_id2label[str(i)] = str(l)
        dict_label2id[str(l)] = str(i)
    return dict_id2label, dict_label2id


def select(data, ids):
    return [data[i] for i in ids]


def shuffle_two(a1, a2):
    """
    Shuffle two lists by same index.
    a1: a list
    a2: an other list
    """
    ran = np.arange(len(a1))
    np.random.shuffle(ran)
    a1_ = [a1[l] for l in ran]
    a2_ = [a2[l] for l in ran]
    return a1_, a2_



def cut_list_by_same_size(data, size):
    """
    data: a list
    size: the chunk_size
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data) - 1) // size + 1)]


def cut_list_by_size_different(data, lengths):
    """
    data: a list
    lengths: the different sizes of cut
    """
    list_block = []
    for l in lengths:
        list_block.append(data[:l])
        data = data[l:]
    return list_block


def cut_by_index(List, indexs):
    """
    List: a list of data
    indexs: a list of indexs
    """
    inds, ind = [], []
    length = len(List)
    for i, l in enumerate(List):
        if l not in indexs:
            ind.append(l)
            if i == length - 1:
                inds.append(ind)
        else:
            if ind != []:
                inds.append(ind)
            ind = []
    return inds


if __name__ == '__main__':
    #
    text = '  总之，伤心，恼火'
    # print(_clean_text(text))
