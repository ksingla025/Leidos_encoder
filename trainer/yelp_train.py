#!/usr/bin/python

#standard python imports
from collections import Counter
import math
import os
import random
import zipfile
import glob
import ntpath
import re
import random
from itertools import compress
import _pickle as cPickle
import pdb
from pathlib import Path
import pdb
import json
import cProfile

#library imports
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn
from sklearn.base import BaseEstimator, TransformerMixin

# external library imports
#from utils.twokenize import *
from model import DocClassifier
from lib.path import *
from lib.util import *
from lib.attention_based_aggregator import *
from lib.batch_data_generators import generate_batch_data_task_yelp

import time                                                

def timeme(method):
    def wrapper(*args, **kw):
        startTime = int(round(time.time() * 1000))
        result = method(*args, **kw)
        endTime = int(round(time.time() * 1000))

        print(endTime - startTime,'ms')
        return result

    return wrapper

class YelpTraining(object):

	def __init__(self,parameter_file='./lib/parameters.json', model_name='yelp_test',epoch=10,
		pre_trained_encoder=None, task_batch_size=25,num_steps=2000):

		self.task_batch_size = task_batch_size
		self.num_steps = num_steps

		self.model_name = model_name

		
		self.params = json.loads(open(parameter_file).read())
		print("paramters file loaded")

		self.coord = tf.train.Coordinator()
		
		self.yelp_data = {}
		self.yelp_data_index = {}

		self.yelp_data_index['train'] = 0
		self.yelp_data['train'] = generate_batch_data_task_yelp(filename=DATA_ID+"yelp_train.p")
		
		self.yelp_data_index['test'] = 0
		self.yelp_data['test'] = generate_batch_data_task_yelp(filename=DATA_ID+"yelp_test.p")

		if pre_trained_encoder == None:
			self.classifier = DocClassifier(embedding_size=64	, sent_aggregator=self.params['sent_aggregator'],
				task_batch_size=self.params['task_batch_size'], valid_size=self.params['valid_size'],
				learning_rate=self.params['learning_rate'], sent_attention_size=self.params['sent_attention_size'],
				doc_attention_size=self.params['doc_attention_size'], sent_embedding_size=self.params['sent_embedding_size'],
				doc_embedding_size=self.params['doc_embedding_size'], sent_lstm_layer=1,
				doc_lstm_layer=self.params['doc_lstm_layer'], leidos_num_classes=self.params['leidos_num_classes'],
				ldcsf_num_classes=self.params['ldcsf_num_classes'], task_learning_rate=self.params['task_learning_rate'],
				multiatt=self.params['multiatt'], model_name=self.model_name, max_length=self.params['max_length'],
				sentsim_learning_rate=self.params['sentsim_learning_rate'], sentsim_batch_size=self.params['sentsim_batch_size'],
				threshold=self.params['threshold'], skipgram_learning_rate=self.params['skipgram_learning_rate'],
				skipgram_batch_size=self.params['skipgram_batch_size'], skipgram_num_sampled=self.params['skipgram_num_sampled'])
			print("DocClassifier initiated !!")

			self.session = tf.Session(graph=self.classifier.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.params['num_threads']))

			self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)

			self.session.run(self.classifier.init_op)

#			self.classifier.var_list_yelp = self.classifier.var_list_yelp + self.classifier.var_list_sentsim

			self.doc_batch = self.classifier.yelp_doc_batch
			self.sentlen_batch = self.classifier.yelp_sentlen_batch
			self.doclen_batch = self.classifier.yelp_doclen_batch
			self.labels_batch = self.classifier.yelp_labels_batch
			self.keep_prob = self.classifier.keep_prob
			self.training_OP_train = self.classifier.yelp_mlp_optimizer_all
			self.cost = self.classifier.yelp_cost
			self.acc = self.classifier.yelp_acc
			self.attention_task = self.classifier.sent_aggregator.attention_task
			self.saver = self.classifier.saver
			self.summary_writer = tf.summary.FileWriter(self.classifier.logs_path, graph=self.classifier.graph)

		else:

			print("Loading pre-trained sentence encoder model")
			self.logs_path = LOGS_PATH+model_name
			self.session = tf.Session()
			self.saver = tf.train.import_meta_graph(pre_trained_encoder+'.meta')
			self.saver.restore(self.session,tf.train.latest_checkpoint('./models/'))

			self.graph = tf.get_default_graph()
			self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)

			self.doc_batch = self.graph.get_tensor_by_name("yelp_document_batch:0")
			self.sentlen_batch = self.graph.get_tensor_by_name("yelp_sentlen_batch:0")
			self.doclen_batch = self.graph.get_tensor_by_name("yelp_doclen_batch:0")
			self.labels_batch = self.graph.get_tensor_by_name("yelp_labels_batch:0")
			self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
			self.training_OP_train = self.graph.get_operation_by_name("Yelp-prediction-SGD/yelp_optimizer_pretrain")
			self.cost = self.graph.get_tensor_by_name("Yelp-prediction-SGD/yelp_cost:0")
			self.acc = self.graph.get_tensor_by_name("Yelp-prediction-SGD/yelp_accuracy:0")
			self.attention_task = self.graph.get_tensor_by_name("Doc-AttentionBasedAggregator/sentattention_vector:0")
			self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

	@timeme
	def _generate_batch_classfication(self,mode='train',task_batch_size=25):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		max_doc_len = []

		for i in range(task_batch_size):

			doc, doclen, sentlen, labels = self.yelp_data[mode][self.yelp_data_index[mode]]

			doc_batch.append(doc)
			doclen_batch.append(doclen)
			sentlen_batch.append(sentlen)
			max_sent_len.append(max(sentlen))
			labels_batch.append(labels)

			self.yelp_data_index[mode] = (self.yelp_data_index[mode] + 1) % len(self.yelp_data[mode])
			if self.yelp_data_index[mode] == 0:
				random.shuffle(self.yelp_data[mode])

		max_doc_len = max(doclen_batch)
		max_sent_len = max(max_sent_len)
		for i in range(task_batch_size):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=max_sent_len, doc_length=max_doc_len, sent_length=sentlen_batch[i])
#		return doc_batch, doclen_batch, sentlen_batch, labels_batch
		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch), np.array(labels_batch)

	def train(self):

#		coord = tf.train.Coordinator()

#		session = tf.Session(graph=self.classifier.graph, config=tf.ConfigProto(
#			intra_op_parallelism_threads=self.params['num_threads']))

#		threads = tf.train.start_queue_runners(sess=session, coord=coord)

#		session.run(self.classifier.init_op)

#		summary_writer = tf.summary.FileWriter(self.classifier.logs_path, graph=self.classifier.graph)

		valid_loss = 999999.0


		valid_doc_batch, valid_doclen_batch, valid_sentlen_batch, valid_labels_batch = self._generate_batch_classfication(mode='test',task_batch_size=1000)
		
		print('starting training')
		for step in range(self.num_steps):
			doc_batch, doclen_batch, sentlen_batch, labels_batch = self._generate_batch_classfication(mode='train')

			print(doc_batch.shape)
			print(doclen_batch.shape)
			print(sentlen_batch.shape)
			print(labels_batch.shape)

			_, cost, accuracy, attention_task = self.session.run([self.training_OP_train, self.cost , self.acc,
				self.attention_task], feed_dict={self.doc_batch: doc_batch,
				self.sentlen_batch: sentlen_batch, self.doclen_batch: doclen_batch,
				self.labels_batch: labels_batch, self.keep_prob : 0.7})
			
			summary_cost = tf.Summary(value=[tf.Summary.Value(tag="DocClassifier-train-cost",
					 simple_value=float(cost))])
			summary_accuracy = tf.Summary(value=[tf.Summary.Value(tag="DocClassifier-train-accuracy",
					 simple_value=float(accuracy))])

			self.summary_writer.add_summary(summary_cost, step)
			self.summary_writer.add_summary(summary_accuracy, step)

			print(attention_task)
			
			if step%300 == 0:

				cost, accuracy = self.session.run([self.cost , self.acc],
					feed_dict={self.doc_batch: valid_doc_batch,self.sentlen_batch: valid_sentlen_batch,
					self.doclen_batch: valid_doclen_batch, self.labels_batch: valid_labels_batch,
					self.keep_prob : 1.0})

				summary_valid_cost = tf.Summary(value=[tf.Summary.Value(tag="DocClassifier-valid-cost",
					 simple_value=float(cost))])

				summary_valid_accuracy = tf.Summary(value=[tf.Summary.Value(tag="DocClassifier-valid-accuracy",
					 simple_value=float(accuracy))])

				self.summary_writer.add_summary(summary_valid_cost, step)
				self.summary_writer.add_summary(summary_valid_accuracy, step)
