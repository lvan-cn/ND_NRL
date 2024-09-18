#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
from glove import Glove
from glove import Corpus
import gensim.utils as utils
from gensim.models import Word2Vec

from sklearn.metrics import accuracy_score
import numpy as np

def pairwise_precision_recall_f1(preds, truths):
    tp = 0
    fp = 0
    fn = 0
    n_samples = len(preds)
    for i in range(n_samples - 1):
        pred_i = preds[i]
        for j in range(i + 1, n_samples):
            pred_j = preds[j]
            if pred_i == pred_j:
                if truths[i] == truths[j]:
                    tp += 1
                else:
                    fp += 1
            elif truths[i] == truths[j]:
                fn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    if tp_plus_fp == 0:
        precision = 0.
    else:
        precision = tp / tp_plus_fp
    if tp_plus_fn == 0:
        recall = 0.
    else:
        recall = tp / tp_plus_fn

    if not precision or not recall:
        f1 = 0.
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

def Kmetric():
    pass


def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def save_emb(glove_mod, filename):
    i = 0
    fn = filename.split('/')[-1]
    embedding_file = '../emb/'+fn
    with open(embedding_file, 'wb') as fout:
        fout.write("%s %s\n" % glove_mod.word_vectors.shape)
        for v in glove_mod.word_vectors:
            #print v
            fout.write("%s %s\n" % (glove_mod.inverse_dictionary[i], ' '.join("%f" % val for val in v)))
            i += 1


def get_emb(glove_mod, filename, dataset):
    # print(glove_mod.dictionary.keys())

    if str(dataset.paper_list[0]) in glove_mod.dictionary.keys():
        D_matrix = glove_mod.word_vectors[glove_mod.dictionary[str(dataset.paper_list[0])]]
    else:
        D_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in dataset.paper_list[1:]:
        if str(idx) in glove_mod.dictionary.keys():
            xx = glove_mod.word_vectors[glove_mod.dictionary[str(idx)]]
        else:
            xx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        D_matrix = np.vstack((D_matrix, xx))
    return D_matrix


def get_emb_walklet(kv, dataset):

    if str(dataset.paper_list[0]) in kv.index2word:
        D_matrix = kv[str(dataset.paper_list[0])]
    else:
        D_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in dataset.paper_list[1:]:
        if str(idx) in kv.index2word:
            xx = kv[str(idx)]
        else:
            xx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        D_matrix = np.vstack((D_matrix, xx))
    return D_matrix



def get_emb_dw(kv, filename, dataset):
    # print(glove_mod.dictionary.keys())
    # xx = kv.vocab.keys()

    if str(dataset.paper_list[0]) in kv.index2word:
        D_matrix = kv[str(dataset.paper_list[0])]
    else:
        D_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in dataset.paper_list[1:]:
        if str(idx) in kv.index2word:
            xx = kv[str(idx)]
        else:
            xx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        D_matrix = np.vstack((D_matrix, xx))
    return D_matrix


def get_emb_n2v(kv, filename, dataset):
    # print(glove_mod.dictionary.keys())

    if str(dataset.paper_list[0]) in kv.index2word:
        D_matrix = kv[str(dataset.paper_list[0])]
    else:
        D_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,]
    for idx in dataset.paper_list[1:]:
        if str(idx) in kv.index2word:
            xx = kv[str(idx)]
        else:
            xx = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ]
        D_matrix = np.vstack((D_matrix, xx))

    return D_matrix


def learn_embeddings(walks, filename, dataset, epoch):
    walks1 = [map(str, walk) for walk in walks] # 映射到str list
    # print(walks)
    embeddings_model = Glove(no_components=20, learning_rate=0.05)
    corpus = Corpus()
    corpus.fit(walks1, window=10)
    embeddings_model.fit(corpus.matrix, epochs=epoch, verbose=True)
    embeddings_model.add_dictionary(corpus.dictionary)
    return embeddings_model


def save_embedding(glove_mod, dataset, filename):
    fn=filename.split('/')[-1]
    embedding_file=open('../emb/'+fn,'w')
    if str(dataset.paper_list[0]) in glove_mod.dictionary.keys():
        D_matrix = glove_mod.word_vectors[glove_mod.dictionary[str(dataset.paper_list[0])]]
    else:
        D_matrix = [0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in dataset.paper_list[1:]:

        if str(idx) in glove_mod.dictionary.keys():
            xx = glove_mod.word_vectors[glove_mod.dictionary[str(idx)]]
        else:
            xx = [0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        D_matrix = np.vstack((D_matrix, xx))
    D_matrix=np.hstack((np.array([range(len(dataset.paper_list))]).T,D_matrix))
    np.savetxt(embedding_file,D_matrix,fmt=''.join(['%i']+[' ']+['%1.5f ']*glove_mod.word_vectors.shape[1]))
    
    












