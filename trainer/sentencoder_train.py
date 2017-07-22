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


class SentEncoderTraining(object):

	def __init__(self, parameter_file='./lib/parameters.json',model_name='encoder_test',
		num_steps=10000,pre_trained=0):

		self.params = json.loads(open(parameter_file).read())
		print("paramters file loaded")

		self.model_name = model_name
		self.num_steps = num_steps

		if pre_trained == 0:

			print("Initiate DocClassifier")
			self.classifier = DocClassifier(embedding_size=self.params['embedding_size'], sent_aggregator=self.params['sent_aggregator'],
				task_batch_size=self.params['task_batch_size'], valid_size=self.params['valid_size'],
				learning_rate=self.params['learning_rate'], sent_attention_size=self.params['sent_attention_size'],
				doc_attention_size=self.params['doc_attention_size'], sent_embedding_size=self.params['sent_embedding_size'],
				doc_embedding_size=self.params['doc_embedding_size'], sent_lstm_layer=self.params['sent_lstm_layer'],
				doc_lstm_layer=self.params['doc_lstm_layer'], leidos_num_classes=self.params['leidos_num_classes'],
				ldcsf_num_classes=self.params['ldcsf_num_classes'], task_learning_rate=self.params['task_learning_rate'],
				multiatt=self.params['multiatt'], model_name=self.model_name, max_length=self.params['max_length'],
				sentsim_learning_rate=self.params['sentsim_learning_rate'], sentsim_batch_size=self.params['sentsim_batch_size'],
				threshold=self.params['threshold'], skipgram_learning_rate=self.params['skipgram_learning_rate'],
				skipgram_batch_size=self.params['skipgram_batch_size'], skipgram_num_sampled=self.params['skipgram_num_sampled'])
			print("DocClassifier initiated !!")

	def train(self):

		# create a session
		coord = tf.train.Coordinator()

		session = tf.Session(graph=self.classifier.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.params['num_threads']))

		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		session.run(self.classifier.init_op)

		summary_writer = tf.summary.FileWriter(self.classifier.logs_path, graph=self.classifier.graph)

		valid_loss = 999999.0

		for step in range(self.num_steps):

			print(step)

			_, loss_val,summary = session.run([self.classifier.skipgram_graph.skip_optimizer,
				self.classifier.skipgram_graph.skip_loss,self.classifier.merged_summary_skip])

			print("skipgram-loss :", loss_val)
				#add  loss summary at step
#				summary_writer.add_summary(summary, step)

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
			
			if step%200 == 0:

				# create batches for sentence similarity task
				valid_sentsimx_batch, valid_sentsimy_batch, valid_sentsimx_len_batch, valid_sentsimy_len_batch,\
				valid_sentsim_labels_batch = session.run([self.classifier.valid_sentsimx_batch,
					self.classifier.valid_sentsimy_batch, self.classifier.valid_sentsimx_len_batch,
					self.classifier.valid_sentsimy_len_batch, self.classifier.valid_sentsim_labels_batch])


				sentsim_cost_mean, sentsim_cost_match_mean, sentsim_cost_mismatch_mean = session.run([self.classifier.cost_mean,
					self.classifier.cost_match_mean, self.classifier.cost_mismatch_mean],feed_dict={self.classifier.train_sentsimx:\
					valid_sentsimx_batch, self.classifier.train_sentsimy: valid_sentsimy_batch, self.classifier.train_sentsimx_len:\
					valid_sentsimx_len_batch, self.classifier.train_sentsimy_len: valid_sentsimy_len_batch,\
					self.classifier.train_sentsim_labels: valid_sentsim_labels_batch, self.classifier.keep_prob : 1.0 })

				if sentsim_cost_mean < valid_loss:
					valid_loss = sentsim_cost_mean
					self.classifier.saver.save(session,self.classifier.model_path)

				summary = tf.Summary(value=[tf.Summary.Value(tag="valid-sentsim-loss",
					simple_value=float(sentsim_cost_mean))])

				summary_match = tf.Summary(value=[tf.Summary.Value(tag="valid-sentsim-match-loss",
					simple_value=float(sentsim_cost_match_mean))])

				summary_mismatch = tf.Summary(value=[tf.Summary.Value(tag="valid-sentsim-mismatch-loss",
					simple_value=float(sentsim_cost_mismatch_mean))])

				summary_writer.add_summary(summary, step)
				summary_writer.add_summary(summary_match, step)
				summary_writer.add_summary(summary_mismatch, step)

