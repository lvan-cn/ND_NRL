#!/usr/bin/env python
# -*- coding: utf-8 -*-

import data_parser
import eval_metric
import os
import time
import multiprocessing as mp
import pandas as pd
# import gensim
import graph_walk
from utility import learn_embeddings

F1_List = []
F1_pre_List=[]
F1_rec_List=[]
F1_pf1_List = []

F1_Max_List = []
F1_Max_List_pre=[]
F1_Max_List_rec=[]
F1_Max_List_pf1 = []

f1_name = {}


def get_file_list(file_dir):
	file_list = []
	for root, dirs, files in os.walk(file_dir):
		file_list.append(files)
	return file_list[0]

def get_median(data):
        data = sorted(data)
        size = len(data)
        if size % 2 == 0:   # 判断列表长度为偶数
            median = (data[size//2]+data[size//2-1])/2
            data[0] = median
        if size % 2 == 1:   # 判断列表长度为奇数
            median = data[(size-1)//2]
            data[0] = median
        return data[0]


def main(filename):
	"""
	pipeline for representation learning for all papers for a given name reference
	"""
	dataset = data_parser.DataSet(filename)
	dataset.reader_arnetminer()
	nx_g = dataset.D_Graph

	g = graph_walk.BasicWalker(nx_g)
	walks = g.simulate_walks(6, 60)
	epoch = 10
	embeddings = learn_embeddings(walks, filename, dataset, epoch=epoch)

	eval_f1 = eval_metric.Evaluator()
	ave_f1, average_pre, average_rec, pf1 = eval_f1.compute_f1(embeddings, filename, dataset)

	F1_Max_List.append(ave_f1)
	F1_Max_List_pre.append(average_pre)
	F1_Max_List_rec.append(average_rec)
	F1_Max_List_pf1.append(pf1)

	print 'avg_f1: ', ave_f1,
	print 'average_pre: ', average_pre
	print 'average_rec: ', average_rec
	print 'average_pf1: ', pf1
	print '\n'


if __name__ == "__main__":

	files_path = "../data/Arnetminer/"

	file_list = get_file_list(files_path)
	file_list = sorted(file_list)
	file_list = file_list[:]
	cnt = 0
	copy_f1_list = []
	copy_pre_list = []
	copy_rec_list = []
	copy_pf1_list = []

	for x in file_list:
		cnt += 1
		filename = files_path + str(x)
		print str(x)
		print "count:" + str(cnt)
		print time.strftime('%H:%M:%S', time.localtime(time.time()))
		F1_Max_List = []
		F1_Max_List_pre = []
		F1_Max_List_rec = []
		F1_Max_List_pf1 = []

		for i in range(1):  
			main(filename)                                                                                           
		f1_name[x]=[]
		f1_name[x].append(get_median(F1_Max_List))
		f1_name[x].append(max(F1_Max_List_pre))
		f1_name[x].append(max(F1_Max_List_rec))

		F1_List.append(max(F1_Max_List))
		F1_pre_List.append(max(F1_Max_List_pre))
		F1_rec_List.append(max(F1_Max_List_rec))
		F1_pf1_List.append(max(F1_Max_List_pf1))

		copy_f1_list.append(max(F1_Max_List))
		copy_pre_list.append(max(F1_Max_List_pre))
		copy_rec_list.append(max(F1_Max_List_rec))
		copy_pf1_list.append(max(F1_Max_List_pf1))
		# print "real time f1:" + str(sum(F1_List) / len(F1_List)) # 所有数据当前的F1的平均

	avg_f1 = sum(copy_f1_list) / len(copy_f1_list)
	avg_pre = sum(copy_pre_list) / len(copy_pre_list)
	avg_rec = sum(copy_rec_list) / len(copy_rec_list)
	avg_pf1 = sum(copy_pf1_list) / len(copy_pf1_list)

	file_list.append('avg')
	copy_f1_list.append(avg_f1)
	copy_pre_list.append(avg_pre)
	copy_rec_list.append(avg_rec)
	copy_pf1_list.append(avg_pf1)

	dataframe = pd.DataFrame({"author": file_list, "macro_f1": copy_f1_list,
							  "pre": copy_pre_list, "rec": copy_rec_list, "K_metric": copy_pf1_list, })

	dataframe.to_csv("res_citeseer/res.csv")
