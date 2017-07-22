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
from model import DocClassifier, get_class_logits
from lib.path import *
from lib.util import *
from lib.attention_based_aggregator import *
from lib.batch_data_generators import generate_train_batch_data_task_leidos,\
generate_test_batch_data_task_leidos, generate_batch_data_task_ldcsf

class LeidosTraining(object):

	def __init__(self,parameter_file='./lib/parameters.json', model_name='leidos_test',epoch=10,
		pre_trained_encoder=None, task_batch_size=25,valid_size=150):

		self.task_batch_size = task_batch_size
		self.valid_size = valid_size

		self.params = json.loads(open(parameter_file).read())
		print("paramters file loaded")
		
		self.model_name = model_name

		self.coord = tf.train.Coordinator()
		self.model_path = MODEL_PATH+model_name

		print("Loading LEIDOS train data")
		self.leidos_data_index = 0
		self.leidos_data = generate_train_batch_data_task_leidos(max_sent_len=50,
			max_doc_size=100)
		print("LEIDOS train data loaded !!")

		print("Loading LEIDOS test data")
		self.leidos_test_data_index = 0
		self.leidos_test_data = generate_test_batch_data_task_leidos(max_sent_len=50,
			max_doc_size=100)
		print("LEIDOS test data loaded !!")

		self.num_steps = len(self.leidos_data)*epoch

		if pre_trained_encoder == None:

			print("Initiate DocClassifier")
			self.classifier = DocClassifier(embedding_size=self.params['embedding_size'], sent_aggregator=self.params['sent_aggregator'],
				task_batch_size=self.params['task_batch_size'], valid_size=self.params['valid_size'],
				learning_rate=self.params['learning_rate'], sent_attention_size=self.params['sent_attention_size'],
				doc_attention_size=self.params['doc_attention_size'], sent_embedding_size=self.params['sent_embedding_size'],
				doc_embedding_size=self.params['doc_embedding_size'], sent_lstm_layer=0,
				doc_lstm_layer=self.params['doc_lstm_layer'], leidos_num_classes=self.params['leidos_num_classes'],
				ldcsf_num_classes=self.params['ldcsf_num_classes'], task_learning_rate=self.params['task_learning_rate'],
				multiatt=self.params['multiatt'], model_name=self.model_name, max_length=self.params['max_length'],
				sentsim_learning_rate=self.params['sentsim_learning_rate'], sentsim_batch_size=self.params['sentsim_batch_size'],
				threshold=self.params['threshold'], skipgram_learning_rate=self.params['skipgram_learning_rate'],
				skipgram_batch_size=self.params['skipgram_batch_size'], skipgram_num_sampled=self.params['skipgram_num_sampled'])
			print("DocClassifier initiated !!")

			self.session = tf.Session(graph=self.classifier.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.params['num_threads']))

			self.threads = tf.train.start_queue_runners(sess=session, coord=coord)

			self.doc_batch = self.classifier.doc_batch
			self.sentlen_batch = self.classifier.sentlen_batch
			self.doclen_batch = self.classifier.doclen_batch
			self.keep_prob = self.classifier.keep_prob
			self.training_OP_train = self.classifier.training_OP_train
			self.loss = self.classifier.loss
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
		
			self.doc_batch = self.graph.get_tensor_by_name("document_batch:0")
			self.sentlen_batch = self.graph.get_tensor_by_name("sentlen_batch:0")
			self.doclen_batch = self.graph.get_tensor_by_name("doclen_batch:0")
			self.labels_batch = self.graph.get_tensor_by_name("labels_batch:0")
			self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
			self.training_OP_train = self.graph.get_operation_by_name("Leidos-DocClassifier-SGD/training_OP_train")
			self.loss = self.graph.get_tensor_by_name("Leidos-DocClassifier-Loss/leidos_loss:0")
			self.predict_leidos = self.graph.get_tensor_by_name("Leidos-Att-Prediction-Layer/predict_leidos:0")
#			self.saver = tf.train.Saver()
			self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)
#			self.saver = self.graph.get_tensor_by_name("model_saver:0")

		

	def _generate_test_batch_leidos_classfication(self):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		for i in range(0,100):

			doc, doclen, sentlen, labels = self.leidos_test_data['en'][self.leidos_test_data_index]
			assert len(sentlen) == 100

			doc_batch.append(doc)
			doclen_batch.append(doclen)
			sentlen_batch.append(sentlen)
			max_sent_len.append(max(sentlen))
			labels_batch.append(labels)

			self.leidos_test_data_index = (self.leidos_test_data_index + 1) % len(self.leidos_test_data['en'])
			if self.leidos_test_data_index == 0:
				random.shuffle(self.leidos_test_data['en'])
		'''
		max_sent_len = max(max_sent_len)
		max_doc_len = max(doclen_batch)
		for i in range(len(doc_batch)):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=max_sent_len, 
				doc_length=max_doc_len, sent_length=sentlen_batch[i])
		'''
		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch), np.array(labels_batch)

	def _generate_batch_leidos_classfication(self):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		for i in range(0,self.valid_size):

			doc, doclen, sentlen, labels = self.leidos_data[self.leidos_data_index]

			assert len(sentlen) == 100

			doc_batch.append(doc)
			doclen_batch.append(doclen)
			sentlen_batch.append(sentlen)
			max_sent_len.append(max(sentlen))
			labels_batch.append(labels)

			self.leidos_data_index = (self.leidos_data_index + 1) % len(self.leidos_data)

			if self.leidos_data_index == 0:

				random.shuffle(self.leidos_data)
		'''
		max_sent_len = max(max_sent_len)
		max_doc_len = max(doclen_batch)
		for i in range(len(doc_batch)):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=max_sent_len, 
				doc_length=max_doc_len, sent_length=sentlen_batch[i])
		'''
		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch), np.array(labels_batch)

	def train(self):

		valid_loss = 999999.0

		doc_batch_valid, doclen_batch_valid, sentlen_batch_valid, labels_batch_valid = self._generate_test_batch_leidos_classfication()

		for step in range(self.num_steps):
			print(step)
			doc_batch, doclen_batch, sentlen_batch, labels_batch = self._generate_batch_leidos_classfication()
#			print("train labels batch", labels_batch)
			'''
			print(doc_batch.shape)
			print(doclen_batch.shape)
			print(sentlen_batch.shape)
			print(labels_batch.shape)
			'''
			print(labels_batch)
			_, leidos_loss, predict_leidos = self.session.run([self.training_OP_train, self.loss, self.predict_leidos],
				feed_dict={self.doc_batch: doc_batch, self.sentlen_batch: sentlen_batch,
				self.doclen_batch: doclen_batch, self.labels_batch: labels_batch,
				self.keep_prob : self.params['keep_prob']})
			print("leidos_classification_loss :",leidos_loss)
			print(predict_leidos)
#			self.summary_writer.add_summary(summary_leidos, step)

			if step%20 == 0:
#					doc_batch, doclen_batch, sentlen_batch, labels_batch = self._generate_test_batch_leidos_classfication()
					leidos_loss, predict_leidos = self.session.run([self.loss,self.predict_leidos], feed_dict={self.doc_batch: doc_batch_valid,
						self.sentlen_batch: sentlen_batch_valid, self.doclen_batch: doclen_batch_valid,
						self.labels_batch: labels_batch_valid, self.keep_prob : 1.0 })


					print(labels_batch_valid.shape)
					print(predict_leidos.shape)
#					predict_leidos = predict_leidos.transpose((2,0, 1))
#					print(predict_leidos.shape)
#					labels_batch_valid = labels_batch_valid.tolist()
					predict_leidos = predict_leidos.tolist()

					print(predict_leidos)
					correct = 0
					total = 0
					for i in range(0,len(labels_batch_valid)):
							if int(labels_batch_valid[i][2]) == int(predict_leidos[i][0]):
								correct = correct + 1
							total =  total + 1
					print("accuracy ",correct/total)

#					print("valid labels_batch",labels_batch)

#					print("validation_loss :",leidos_loss)
#					print("predict_leidos :",predict_leidos)

					summary = tf.Summary(value=[tf.Summary.Value(tag="DocClassifier-validloss",
					 simple_value=float(leidos_loss))])

					self.summary_writer.add_summary(summary, step)

					if leidos_loss < valid_loss:
						valid_loss = leidos_loss
						self.saver.save(self.session,self.model_path)


