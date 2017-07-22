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


class LdcsfTraining(object):

	def __init__(self,pre_trained_encoder='./models/leidos_test', parameter_file='./lib/parameters.json',
		model_name='ldcsf_test',epoch=10, task_batch_size=5, valid_size=50):

		self.task_batch_size = task_batch_size
		self.valid_size = valid_size

		self.params = json.loads(open(parameter_file).read())
		print("paramters file loaded")

		self.model_path = MODEL_PATH+model_name

		self.ldcsf_data = {}
		self.ldcsf_data_index = {}

		print("Loading LDC SF train data")
		self.ldcsf_data_index['train'] = 0
		self.ldcsf_data['train'] = generate_batch_data_task_ldcsf(filename=DATA_ID+"sec_pilot_train.p",
			max_sent_len=50, max_doc_size=100)
		print("LDC SF train data loaded !!")

		print("Loading LDC SF test data")
		self.ldcsf_data_index['test'] = 0
		self.ldcsf_data['test'] = generate_batch_data_task_ldcsf(filename=DATA_ID+"sec_pilot_eval.p",
			max_sent_len=50, max_doc_size=100)
		print("LDC SF test data loaded !!")

		self.num_steps = len(self.ldcsf_data['train'])*epoch

		print("Loading pre-trained document encoder model")
		self.logs_path = LOGS_PATH+model_name
		self.session = tf.Session()
		self.saver = tf.train.import_meta_graph(pre_trained_encoder+'.meta')
		self.saver.restore(self.session,tf.train.latest_checkpoint('./models/'))
		self.coord = tf.train.Coordinator()
		self.graph = tf.get_default_graph()
		self.threads = tf.train.start_queue_runners(sess=self.session, coord=self.coord)
		
		self.doc_batch = self.graph.get_tensor_by_name("document_batch:0")
		self.sentlen_batch = self.graph.get_tensor_by_name("sentlen_batch:0")
		self.doclen_batch = self.graph.get_tensor_by_name("doclen_batch:0")
		self.labels_batch = self.graph.get_tensor_by_name("labels_batch:0")
		self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
		self.training_ldcsf_train = self.graph.get_operation_by_name("Ldcsf-DocClassifier-SGD/training_ldcsf_train")
		self.ldcsf_loss = self.graph.get_tensor_by_name("Ldcsf-DocClassifier-Loss/loss_ldcsf_reduce_mean:0")
		self.predict_ldcsf = self.graph.get_tensor_by_name("Ldcsf-Att-Prediction-Layer/predict_ldcsf_threshold:0")
		self.summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

		self.saver = tf.train.Saver()

	def _generate_batch_ldcsf_classfication(self, batch_size=5, mode='test'):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		for i in range(0,batch_size):

			doc, doclen, sentlen, labels = self.ldcsf_data[mode][self.ldcsf_data_index[mode]]

			assert len(sentlen) == 100

			doc_batch.append(doc)
			doclen_batch.append(doclen)
			sentlen_batch.append(sentlen)
			max_sent_len.append(max(sentlen))
			labels_batch.append(labels)

			self.ldcsf_data_index[mode] = (self.ldcsf_data_index[mode] + 1) % len(self.ldcsf_data[mode])

			if self.ldcsf_data_index[mode] == 0:

				random.shuffle(self.ldcsf_data[mode])

		max_sent_len = max(max_sent_len)
		max_doc_len = max(doclen_batch)
		for i in range(len(doc_batch)):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=max_sent_len, 
				doc_length=max_doc_len, sent_length=sentlen_batch[i])

		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch), np.array(labels_batch)

	def train(self):

		valid_loss = 999999.0
		for step in range(self.num_steps):

			doc_batch, doclen_batch, sentlen_batch, labels_batch =\
				self._generate_batch_ldcsf_classfication(batch_size=self.task_batch_size, mode='train')

			_, ldcsf_loss, predict_ldcsf=\
			self.session.run([self.training_ldcsf_train, self.ldcsf_loss,
				self.predict_ldcsf], feed_dict={self.doc_batch: doc_batch,
					self.sentlen_batch: sentlen_batch, self.doclen_batch: doclen_batch,
					self.labels_batch: labels_batch, self.keep_prob : self.params['keep_prob'] })

			print("ldcsf classification loss",ldcsf_loss)

			if step % 100 == 0:
				doc_batch, doclen_batch, sentlen_batch, labels_batch =\
				self._generate_batch_ldcsf_classfication(batch_size=self.task_batch_size, mode='test')

				ldcsf_loss, predict_ldcsf = self.session.run([self.ldcsf_loss, self.predict_ldcsf],
					feed_dict={self.doc_batch: doc_batch, self.sentlen_batch: sentlen_batch,
					self.doclen_batch: doclen_batch, self.labels_batch: labels_batch,
					self.keep_prob : 1.0 })

				if ldcsf_loss < valid_loss:
						valid_loss = ldcsf_loss

				print("ldcsf validation loss",ldcsf_loss)

			self.saver.save(self.session,self.model_path)






