#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from xmeans import XMeans
import numpy as np
from utility import save_emb, get_emb, save_embedding, purity_score
# import hdbscan

from sklearn.metrics import recall_score, f1_score, precision_score,confusion_matrix,accuracy_score
from sklearn.metrics import *

# from imblearn.metrics import geometric_mean_score
from math import sqrt


class Evaluator():
    @staticmethod
    def compute_f1(glove, filename, dataset):
        """
        perform Hierarchy Clustering on doc embedding matrix
        for name disambiguation
        use cluster-level mean F1 for evaluation
        """
        D_matrix = get_emb(glove, filename, dataset)
        save_emb(glove, filename)
        save_embedding(glove, dataset, filename)
        
        true_cluster_size = len(set(dataset.label_list))
        print('real author number: ', true_cluster_size)
        y_pred = DBSCAN(eps=1.5, min_samples=2).fit_predict(D_matrix)

        true_label_dict = {}
        for idx, true_lbl in enumerate(dataset.label_list):
            if true_lbl not in true_label_dict:
                true_label_dict[true_lbl] = [idx]
            else:
                true_label_dict[true_lbl].append(idx)  
        # print(true_label_dict)

        predict_label_dict = {}
        for idx, pred_lbl in enumerate(y_pred):
            if pred_lbl not in predict_label_dict:
                predict_label_dict[pred_lbl] = [idx]
            else:
                predict_label_dict[pred_lbl].append(idx)

        # print(predict_label_dict)
        # compute cluster-level F1
        # let's denote C(r) as clustering result and T(k) as partition (ground-truth)
        # construct r * k contingency table for clustering purpose

        r_k_table = []
        for v1 in predict_label_dict.values():       # predict label:key  value: 论文 id
            k_list = []
            for v2 in true_label_dict.values():
                N_ij = len(set(v1).intersection(v2))
                k_list.append(N_ij)
            r_k_table.append(k_list)
        r_k_matrix = np.array(r_k_table)
        r_num = int(r_k_matrix.shape[0])

        N = float(len(dataset.label_list))
        N_ttpp = 0
        for T in true_label_dict.values():  # 每个T都是一个List
            for P in predict_label_dict.values():
                l = len(set(T).intersection(P))
                N_ttpp += (l * l) / float(len(T))
        N_ttpp = N_ttpp / N
        AAP = N_ttpp

        N_ttpp2 = 0
        for T in true_label_dict.values():
            for P in predict_label_dict.values():
                l = len(set(P).intersection(T))
                N_ttpp2 += (l * l) / float(len(P))
        N_ttpp2 = N_ttpp2 / N
        ACP = N_ttpp2
        K_metric = sqrt(AAP * ACP)


        # compute F1 for each row C_i
        sum_f1 = 0.0
        sum_pre = 0.0
        sum_rec = 0.0
        sum_fmi = 0.0

        for row in range(0, r_num):
            row_sum = np.sum(r_k_matrix[row, :])
            if row_sum != 0:
                max_col_index = np.argmax(r_k_matrix[row, :])
                row_max_value = r_k_matrix[row, max_col_index]
                prec = float(row_max_value) / row_sum
                col_sum = np.sum(r_k_matrix[:, max_col_index])
                rec = float(row_max_value) / col_sum
                row_f1 = float(2 * prec * rec) / (prec + rec)

                fmi = float(row_max_value) / sqrt(row_sum * col_sum)

                sum_f1 += row_f1
                sum_pre += prec
                sum_rec += rec
                sum_fmi += fmi

        average_f1 = float(sum_f1) / r_num
        average_pre = float(sum_pre) / r_num
        average_rec = float(sum_rec) / r_num
        average_fmi = float(sum_fmi) / r_num
        return average_f1, average_pre, average_rec, K_metric

