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


class Training(object):

	def __init__(self,parameter_file,model_name='test', num_steps=10000, leidos_start=200,ldcsf_start=300):

		self.model_name = model_name
		self.num_steps = num_steps
		self.leidos_start = leidos_start
		self.ldcsf_start = ldcsf_start

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

		self.params = json.loads(open(parameter_file).read())
		print("paramters file loaded")

		self.task_batch_size = self.params['task_batch_size']

		print("Initiate DocClassifier")
		self.classifier = DocClassifier(embedding_size=self.params['embedding_size'], sent_aggregator=self.params['sent_aggregator'],
			task_batch_size=self.params['task_batch_size'], valid_size=self.params['valid_size'],
			learning_rate=self.params['learning_rate'], sent_attention_size=self.params['sent_attention_size'],
			doc_attention_size=self.params['doc_attention_size'], sent_embedding_size=self.params['sent_embedding_size'],
			doc_embedding_size=self.params['doc_embedding_size'], sent_lstm_layer=self.params['sent_lstm_layer'],
			doc_lstm_layer=self.params['doc_lstm_layer'], leidos_num_classes=self.params['leidos_num_classes'],
			ldcsf_num_classes=self.params['ldcsf_num_classes'], task_learning_rate=self.params['task_learning_rate'],
			multiatt=self.params['multiatt'], model_name=self.model_name, max_length=self.params['max_length'],
			sentsim_learning_rate=self.params['sentsim_learning_rate'], sentsim_batch_size=self.params['sentsim_batch_size'])
		print("DocClassifier initiated !!")

	def _generate_test_batch_leidos_classfication(self):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		for i in range(0,self.task_batch_size):

			doc, doclen, sentlen, labels = self.leidos_test_data['es'][self.leidos_test_data_index]
			assert len(sentlen) == 100

			doc_batch.append(doc)
			doclen_batch.append(doclen)
			sentlen_batch.append(sentlen)
			max_sent_len.append(max(sentlen))
			labels_batch.append(labels)

			self.leidos_test_data_index = (self.leidos_test_data_index + 1) % len(self.leidos_test_data['es'])
			if self.leidos_test_data_index == 0:
				random.shuffle(self.leidos_test_data['es'])

		max_sent_len = max(max_sent_len)
		max_doc_len = max(doclen_batch)
		for i in range(len(doc_batch)):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=max_sent_len, 
				doc_length=max_doc_len, sent_length=sentlen_batch[i])
		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch), np.array(labels_batch)

	def _generate_batch_leidos_classfication(self):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		for i in range(0,self.task_batch_size):

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

		max_sent_len = max(max_sent_len)
		max_doc_len = max(doclen_batch)
		for i in range(len(doc_batch)):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=max_sent_len, 
				doc_length=max_doc_len, sent_length=sentlen_batch[i])

		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch), np.array(labels_batch)

	def _generate_batch_ldcsf_classfication(self, mode='test'):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		labels_batch = []
		max_sent_len = []
		for i in range(0,self.task_batch_size):

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
		
		# create a session
		coord = tf.train.Coordinator()

		session = tf.Session(graph=self.classifier.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.params['num_threads']))

		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		session.run(self.classifier.init_op)

		summary_writer = tf.summary.FileWriter(self.classifier.logs_path, graph=self.classifier.graph)

		for step in range(self.num_steps):

			print(step)
			
			if step > self.ldcsf_start:
				doc_batch, doclen_batch, sentlen_batch, labels_batch =\
				self._generate_batch_ldcsf_classfication(mode='train')

				print(labels_batch.shape)
				_, loss_ldcsf, predictions_ldcsf, predict_ldcsf, logit_ldcsf, summary_ldcsf =\
				session.run([self.classifier.training_ldcsf_train, self.classifier.loss_ldcsf,
					self.classifier.predictions_ldcsf, self.classifier.predict_ldcsf, self.classifier.logit_ldcsf,
					self.classifier.merged_summary_ldcsf_classification], feed_dict={self.classifier.doc_batch: doc_batch,
					self.classifier.sentlen_batch: sentlen_batch, self.classifier.doclen_batch: doclen_batch,
					self.classifier.labels_batch: labels_batch, self.classifier.keep_prob : self.params['keep_prob'] })

				print("ldcsf_classification_loss :",loss_ldcsf)
				print("ldcsf logits :",logit_ldcsf)
				print("ldcsf_predictions :", predictions_ldcsf)
				print("ldcsf_predict :", predict_ldcsf)
				print("ldcsf_predictions :", np.argmax(predictions_ldcsf, 1))


				predict_ldcsf = session.run([self.classifier.predict_ldcsf], feed_dict={self.classifier.doc_batch: doc_batch,
				self.classifier.sentlen_batch: sentlen_batch, self.classifier.doclen_batch: doclen_batch,
				self.classifier.keep_prob : 1.0 })

				print("batch predict", predict_ldcsf)
				summary_writer.add_summary(summary_ldcsf, step)

				if step%100 == 0:
					doc_batch, doclen_batch, sentlen_batch, labels_batch = self._generate_batch_ldcsf_classfication(mode='test')

					print(labels_batch.shape)
					loss_ldcsf, summary = session.run([self.classifier.loss_ldcsf,
						self.classifier.merged_summary_ldcsf_classification], feed_dict={self.classifier.doc_batch: doc_batch,
						self.classifier.sentlen_batch: sentlen_batch, self.classifier.doclen_batch: doclen_batch,
						self.classifier.labels_batch: labels_batch, self.classifier.keep_prob : self.params['keep_prob'] })

					summary = tf.Summary(value=[tf.Summary.Value(tag="LDCSF-DocClassifier-validloss",
					 simple_value=float(loss_ldcsf))])


			
			elif step > self.leidos_start:
				doc_batch, doclen_batch, sentlen_batch, labels_batch = self._generate_batch_leidos_classfication()

				'''
				print(doc_batch.shape)
				print(doclen_batch.shape)
				print(sentlen_batch.shape)
				print(labels_batch.shape)
				'''
				_, leidos_loss, summary_leidos = session.run([self.classifier.training_OP_train, self.classifier.loss,
					self.classifier.merged_summary_leiods_classification], feed_dict={self.classifier.doc_batch: doc_batch,
					self.classifier.sentlen_batch: sentlen_batch, self.classifier.doclen_batch: doclen_batch,
					self.classifier.labels_batch: labels_batch, self.classifier.keep_prob : self.params['keep_prob'] })

				print("leidos_classification_loss :",leidos_loss)
				summary_writer.add_summary(summary_leidos, step)

				if step%100 == 0:
					doc_batch, doclen_batch, sentlen_batch, labels_batch = self._generate_test_batch_leidos_classfication()
					leidos_loss, summary = session.run([self.classifier.loss,
						self.classifier.merged_summary_leiods_classification], feed_dict={self.classifier.doc_batch: doc_batch,
						self.classifier.sentlen_batch: sentlen_batch, self.classifier.doclen_batch: doclen_batch,
						self.classifier.labels_batch: labels_batch, self.classifier.keep_prob : self.params['keep_prob'] })

					summary = tf.Summary(value=[tf.Summary.Value(tag="DocClassifier-validloss",
					 simple_value=float(leidos_loss))])

					summary_writer.add_summary(summary, step)




			else:

				# create batches for sentence similarity task
				train_sentsimx_batch, train_sentsimy_batch, train_sentsimx_len_batch, train_sentsimy_len_batch,\
				train_sentsim_labels_batch = session.run([self.classifier.train_sentsimx_batch,
					self.classifier.train_sentsimy_batch, self.classifier.train_sentsimx_len_batch,
					self.classifier.train_sentsimy_len_batch, self.classifier.train_sentsim_labels_batch])

				#run the training step
				_, sentsim_cost_mean, sentsim_cost_match_mean, sentsim_cost_mismatch_mean, summary_sentsim =\
				session.run([self.classifier.sentsim_optimizer_train, self.classifier.cost_mean, self.classifier.cost_match_mean,
					self.classifier.cost_mismatch_mean, self.classifier.merged_summary_sentsim], feed_dict={self.classifier.train_sentsimx:\
					train_sentsimx_batch, self.classifier.train_sentsimy: train_sentsimy_batch, self.classifier.train_sentsimx_len:\
					train_sentsimx_len_batch, self.classifier.train_sentsimy_len: train_sentsimy_len_batch,\
					self.classifier.train_sentsim_labels: train_sentsim_labels_batch, self.classifier.keep_prob : self.params['keep_prob'] })

				print("sentsim_loss",sentsim_cost_mean)
				summary_writer.add_summary(summary_sentsim, step)

				if step%100 == 0:

					# create batches for sentence similarity task
					valid_sentsimx_batch, valid_sentsimy_batch, valid_sentsimx_len_batch, valid_sentsimy_len_batch,\
					valid_sentsim_labels_batch = session.run([self.classifier.valid_sentsimx_batch,
						self.classifier.valid_sentsimy_batch, self.classifier.valid_sentsimx_len_batch,
						self.classifier.valid_sentsimy_len_batch, self.classifier.valid_sentsim_labels_batch])


					sentsim_cost_mean, sentsim_cost_match_mean, sentsim_cost_mismatch_mean = session.run([self.classifier.cost_mean,
						self.classifier.cost_match_mean, self.classifier.cost_mismatch_mean],feed_dict={self.classifier.train_sentsimx:\
						valid_sentsimx_batch, self.classifier.train_sentsimy: valid_sentsimy_batch, self.classifier.train_sentsimx_len:\
						valid_sentsimx_len_batch, self.classifier.train_sentsimy_len: valid_sentsimy_len_batch,\
						self.classifier.train_sentsim_labels: valid_sentsim_labels_batch, self.classifier.keep_prob : self.params['keep_prob'] })

					summary = tf.Summary(value=[tf.Summary.Value(tag="valid-sentsim-loss",
					 simple_value=float(sentsim_cost_mean))])

					summary_match = tf.Summary(value=[tf.Summary.Value(tag="valid-sentsim-match-loss",
					 simple_value=float(sentsim_cost_match_mean))])

					summary_mismatch = tf.Summary(value=[tf.Summary.Value(tag="valid-sentsim-mismatch-loss",
					 simple_value=float(sentsim_cost_mismatch_mean))])
					
					summary_writer.add_summary(summary, step)
					summary_writer.add_summary(summary_match, step)
					summary_writer.add_summary(summary_mismatch, step)					



			self.classifier.saver.save(session,self.classifier.model_path)