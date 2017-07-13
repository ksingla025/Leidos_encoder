#!/usr/bin/python

''' Author : Karan Singla, Dogan Can '''

''' main file for training word embeddings and get sentence embeddings	'''


#standard python imports
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

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
from lib.path import *
from lib.util import *
from lib.attention_based_aggregator import *
from lib.batch_data_generators import generate_train_batch_data_task_leidos,\
generate_test_batch_data_task_leidos, generate_batch_data_task_ldcsf
############### Utility Functions ####################
def get_class_logits(tf_train_dataset, embedding_size=100, name="test"):
    weights = tf.Variable(tf.truncated_normal([embedding_size, 1]), name=name+"_weights")
    biases = tf.Variable(tf.zeros([1]), name=name+"_bias")
    logits = tf.matmul(tf_train_dataset, weights) + biases
    return weights, biases, logits


class DocClassifier(BaseEstimator, TransformerMixin):

	def __init__(self,embedding_size=100, sent_aggregator=None, task_batch_size=5, valid_size=10,
		learning_rate=0.01, sent_attention_size=100, doc_attention_size=100, sent_embedding_size=100,
		doc_embedding_size=100, sent_lstm_layer=1, doc_lstm_layer=1, leidos_num_classes=50,
		ldcsf_num_classes=12,task_learning_rate=.01, multiatt=True, model_name='test_leidos',
		max_length=50,sentsim_learning_rate=0.01, sentsim_batch_size=20, num_threads=20):

		#set parameters
		self.embedding_size = embedding_size
		self.task_batch_size = task_batch_size
		self.valid_size = valid_size
		self.learning_rate = .01
		self.sent_attention_size = sent_attention_size
		self.doc_attention_size = doc_attention_size
		self.sent_embedding_size = sent_embedding_size
		self.doc_embedding_size = doc_embedding_size
		self.sent_lstm_layer = sent_lstm_layer
		self.doc_lstm_layer = doc_lstm_layer
		self.leidos_num_classes = leidos_num_classes
		self.ldcsf_num_classes = ldcsf_num_classes
		self.multiatt = multiatt
		self.task_learning_rate = task_learning_rate
		self.model_path = MODEL_PATH+model_name
		self.sent_lstm_layer = sent_lstm_layer
		self.doc_lstm_layer = doc_lstm_layer
		self.max_length = max_length
		self.sentsim_learning_rate = sentsim_learning_rate
		self.logs_path = LOGS_PATH+model_name
		self.sentsim_batch_size = sentsim_batch_size

		self._build_dictionaries()

		self.graph = tf.Graph()

		self.doc_classifier_graph()

	def _build_dictionaries(self):

		print("Loading Data Files")

		self.dictionary = cPickle.load(open(DATA_ID+"dictionary.p", 'rb'))
		self.reverse_dictionary = cPickle.load(open(DATA_ID+"reverse_dictionary.p", 'rb'))
		
		print("dictionaries loaded")

		self.vocabulary_size = len(self.dictionary.keys())

	def _initiate_sentence_encoder(self):
		#do a embedding lookup for each word in the document
		self.doc_embed = tf.nn.embedding_lookup(self.embeddings, self.doc_batch)

		#we unstack and pick only one document to initiate the sentence encoder
		self.doc_embed_0 = self.doc_embed[0]
#		self.doc_embed_0 = tf.unstack(self.doc_embed)[0]
			
		'''
		we unstack sentlen to get sentence lengths of a single document
		this is because sentence aggregator/aggregator class can only
		take list of sequences and not list of list of sequences
		'''
		self.sentlen_0 = self.sentlen_batch[0]
			
		self.sent_aggregator = Aggregator(sequence_length=self.sentlen_0,
			embedding_size=self.embedding_size, attention_size=self.sent_attention_size,
			embed = self.doc_embed_0, n_hidden=self.sent_embedding_size, lstm_layer=self.sent_lstm_layer,
			keep_prob=self.keep_prob,idd='sent')
		self.sent_aggregator.init_attention_aggregator()

		self.var_list_sentsim = self.var_list_sentsim + self.sent_aggregator.aggregator_variables
		self.var_list_leidos = self.var_list_sentsim + self.sent_aggregator.aggregator_variables

	def _initiate_document_encoder(self):

		self.document_aggregator = DocAggregator(embedding_size=self.embedding_size,
			sent_embedding_size=self.sent_embedding_size, sent_attention_size=self.sent_attention_size,
			doc_attention_size=self.doc_attention_size, doc_embedding_size=self.doc_embedding_size,
			sent_aggregator=self.sent_aggregator, lstm_layer=self.doc_lstm_layer, keep_prob=self.keep_prob,
			idd='doc', multiatt=self.multiatt)
		self.document_aggregator.initiate_doc_attention_aggregator(doc_embed=self.doc_embed,
			sent_len=self.sentlen_batch, doc_len=self.doclen_batch, num_class=self.leidos_num_classes)

		self.var_list_sentsim = self.var_list_sentsim + self.document_aggregator.doc_attention_aggregator.aggregator_variables
		self.var_list_leidos = self.var_list_sentsim + self.document_aggregator.doc_attention_aggregator.aggregator_variables

	def _get_sentence_encodings(self):

		self.embedx = tf.nn.embedding_lookup(self.embeddings, self.train_sentsimx)
		self.embedx = tf.nn.dropout(self.embedx, self.keep_prob)

		self.embedy = tf.nn.embedding_lookup(self.embeddings, self.train_sentsimy)
		self.embedy = tf.nn.dropout(self.embedy, self.keep_prob)

		# if using lstm layer
		if self.sent_lstm_layer == 1:
				
			contextx = self.sent_aggregator.calculate_attention_with_lstm(self.embedx,self.train_sentsimx_len)
			contextx = tf.nn.dropout(contextx, self.keep_prob)

			contexty = self.sent_aggregator.calculate_attention_with_lstm(self.embedx,self.train_sentsimx_len)
			contexty = tf.nn.dropout(contexty, self.keep_prob)

		# if no lstm layer
		if self.sent_lstm_layer == 0:

			contextx = self.sent_aggregator.calculate_attention(self.embedx)
			contextx = tf.nn.dropout(contextx, self.keep_prob)

			contexty = self.sent_aggregator.calculate_attention(self.embedy)
			contexty = tf.nn.dropout(contexty, self.keep_prob)

		return contextx, contexty

	def doc_classifier_graph(self):
		'''
		this is the main network function for document classifier
		Features ( can use ):
		1. LSTM (with attention) based encoding of sentences
		2. Pre-load a sentence encoder
		3. LSTM (with attention) based encoding of documents from sentences
		4. Classifier
		'''

		with self.graph.as_default(), tf.device('/cpu:0'):
			'''
			doc_batch is [batch_size,max_doc_size,max_sent_len]
			sentlen_batch contains original length of each sentence in a document
			doc_len contains original number of sentences in a document
			labels batch contains labels vectors, where classes present are 1
			'''
			################### define all placeholders #####################

			##DocClassifier placeholders
			self.doc_batch = tf.placeholder(tf.int32, [None,None,None], name='document_batch')
			self.sentlen_batch = tf.placeholder(tf.int32, [None,None], name='sentlen_batch')
			self.doclen_batch = tf.placeholder(tf.int32, [None], name='doclen_batch')			
			####Leidos labels
			self.labels_batch = tf.placeholder(tf.float32, [None,50], name='labels_batch')

			self.ldcsf_indexes = list(range(self.ldcsf_num_classes))

			#keep first 15 of labels vector for ldc sf
			#default axis for gather_axis is -1
			self.labels_batch_ldcsf = gather_axis(self.labels_batch,self.ldcsf_indexes, axis=-1)
			####SF labels
#			self.sf_labels_batch = tf.placeholder(tf.float32, [None,None], name='ldcsf_labels_batch')

			##SentSim placeholders
			### training batch extractor
			self.train_sentsimx_batch, self.train_sentsimy_batch, self.train_sentsimx_len_batch,\
			self.train_sentsimy_len_batch,self.train_sentsim_labels_batch =\
			self.input_pipeline_sentsim(filenames=[DATA_BATCH+'sentsim.csv'],
				batch_size=self.sentsim_batch_size)

			# validation batch extractor
			self.valid_sentsimx_batch, self.valid_sentsimy_batch, self.valid_sentsimx_len_batch,\
			self.valid_sentsimy_len_batch,self.valid_sentsim_labels_batch =\
			self.input_pipeline_sentsim(filenames=[DATA_BATCH+'valid_es_en.csv'],
				batch_size=self.sentsim_batch_size)


			#### sent encoder parallel sentences
			self.train_sentsimx = tf.placeholder(tf.int32, [None,None], name='sentsim-inputx')
			self.train_sentsimy = tf.placeholder(tf.int32, [None,None], name='sentsim-inputy')
			self.train_sentsimx_len = tf.placeholder(tf.int32, [None], name='sentsim-inputx_len')
			self.train_sentsimy_len = tf.placeholder(tf.int32, [None], name='sentsim-inputy_len')
			self.train_sentsim_labels = tf.placeholder(tf.float32, [None, 1], name='sentsim-outlabel')

			#######---------------------------*------------------------#########

			self.var_list_sentsim = []
			self.var_list_leidos = []
			self.var_list_ldcsf = []
			#step to mamnage decay
			self.global_step = tf.Variable(0, trainable=False)

			#parameter to control droupput, keep_prob-0.7, means it keeps 0.7 of the data
			self.keep_prob = tf.placeholder("float",name='keep_prob')

			#initiate the embeddings for each word in the vocabulary
			self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size,
				self.embedding_size], -1.0, 1.0), name='word_embeddings')

			self.var_list_sentsim.append(self.embeddings)



			####----initiate sentence encoder and document encoders with attention----#####


			'''
			we initiate the aggregator for encoding sentences
			use lstm_layer=1 for if using lstm layer else 0
			it generates self.sent_aggregator
			'''
			with tf.variable_scope('AttentionBasedAggregator') as scope:
				self._initiate_sentence_encoder()

				'''
				we initiate the aggregator for encoding documents with
				the doc_batch. here also keep lstm_layer, if using lstm
				layer at the document level
				it creates self.doc_aggregator
				'''
				self._initiate_document_encoder()

				self.doc_embed = tf.nn.embedding_lookup(self.embeddings, self.doc_batch)
				self.document_vector = self.document_aggregator.calculate_document_vector(self.doc_embed,
					self.sentlen_batch, self.doclen_batch)


				if self.multiatt == True:
#					self.document_vector = tf.transpose(self.document_vector, perm=[1,0,2])
#					self.document_vector = tf.contrib.layers.flatten(self.document_vector)

					self.document_vector_ldcsf = tf.transpose(self.document_vector, perm=[1,0,2])
					self.document_vector_ldcsf = tf.contrib.layers.flatten(self.document_vector_ldcsf)

					self.document_vector = tf.unstack(self.document_vector, axis=0,name='document-vectorss')
				################ sentence similarity objective ################
				'''
				get embeddings for x and y input sentence for sentence similarity objective
				it will generate self.contextx, self.contexty for self.train_sentsimx &
				self.train_sentsimy
				self.contextx = self.sent_aggregator.calculate_attention_with_lstm(self.embedx,self.train_sentsimx_len)
				where embedx is emembedding lookup on self.train_sentsimx
				'''
				self.contextx, self.contexty = self._get_sentence_encodings()




			with tf.name_scope('SentSim-Loss'):
				self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean = loss(self.contextx,
					self.contexty, self.train_sentsim_labels)
				tf.summary.scalar("task_loss_divide", self.cost_mean, collections=['sentsim'])
				tf.summary.scalar("task_loss_match_divide", self.cost_match_mean, collections=['sentsim'])
				tf.summary.scalar("task_loss_mismatch_divide", self.cost_mismatch_mean, collections=['sentsim'])
			
			with tf.name_scope('Task-SGD'):
				self.sent_learning_rate = tf.train.exponential_decay(self.sentsim_learning_rate,
					self.global_step, 50000, 0.98, staircase=True)


				self.sentsim_optimizer = tf.train.GradientDescentOptimizer(self.sent_learning_rate)

				self.sentsim_optimizer_train = self.sentsim_optimizer.minimize(self.cost_mean,
					var_list=self.var_list_sentsim)

			########--------------------------*-------------------------########

			############## Leidos classification objective #################

			'''
			this is the used after the document encoder has been initialized
			and being used for training
			'''
			
#				self.document_vector = tf.unstack(self.document_vector, axis=0)
#				print(self.document_vector)

			#if using multiatt=1 for document encoder

			#if using multiatt=1 for document encoder
			with tf.variable_scope('Leidos-Att-Prediction-Layer') as scope:
				#### prediction layer
				pred_weights = []
				pred_bias = []
				logits = []
				for i in range(0,len(self.document_vector)):
					w_0, b_0, logits_0 = get_class_logits(self.document_vector[i],
						self.doc_embedding_size, name="pred"+str(i))

					self.var_list_leidos.append(w_0)
					self.var_list_leidos.append(b_0)
					pred_weights.append(w_0)
					pred_bias.append(b_0)
					logits.append(logits_0)

				logit_leidos = tf.concat(logits,1)
			
			with tf.variable_scope('Ldcsf-Att-Prediction-Layer') as scope:

				if self.multiatt == True:
					pred_weights_ldcsf = tf.Variable(tf.truncated_normal([self.doc_embedding_size*self.leidos_num_classes, self.ldcsf_num_classes]),
						name="pred_weights_ldcsf")
				else:
					pred_weights_ldcsf = tf.Variable(tf.truncated_normal([self.doc_embedding_size, self.ldcsf_num_classes]),
						name="pred_weights_ldcsf")
				pred_bias_ldcsf = tf.Variable(tf.zeros([self.ldcsf_num_classes]), name="pred_bias_ldcsf")

				self.var_list_ldcsf.append(pred_weights_ldcsf)
				self.var_list_ldcsf.append(pred_bias_ldcsf)

				self.logit_ldcsf = tf.matmul(self.document_vector_ldcsf, pred_weights_ldcsf) + pred_bias_ldcsf
				self.predictions_ldcsf = tf.nn.sigmoid(self.logit_ldcsf)
				self.predict_ldcsf = tf.where(self.predictions_ldcsf > 0.5, tf.ones_like(self.predictions_ldcsf),
					tf.zeros_like(self.predictions_ldcsf),name='predict_ldcsf_1')

			with tf.name_scope('Leidos-DocClassifier-Loss'):

				self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit_leidos,
					labels=self.labels_batch))
				tf.summary.scalar("leiods_classifcation_loss", self.loss, collections=['leidos-classification'])
	
			with tf.name_scope('Leidos-DocClassifier-SGD'):
				self.learning_rate = tf.train.exponential_decay(self.task_learning_rate, self.global_step,
					50000, 0.98, staircase=True)
				
				# OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
				self.training_OP = tf.train.GradientDescentOptimizer(self.learning_rate)

				self.var_list_leidos = self.var_list_leidos[1:]

				print(self.var_list_leidos)
				self.training_OP_train = self.training_OP.minimize(self.loss,
					var_list=self.var_list_leidos)

			with tf.name_scope('Ldcsf-DocClassifier-Loss'):

				self.loss_ldcsf = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_ldcsf,
					labels=self.labels_batch_ldcsf), name='loss_ldcsf_reduce_mean')
				tf.summary.scalar("ldcsf_classifcation_loss", self.loss_ldcsf, collections=['ldcsf-classification'])

			with tf.name_scope('Ldcsf-DocClassifier-SGD'):
				self.learning_rate_ldcsf = tf.train.exponential_decay(self.task_learning_rate, self.global_step,
					50000, 0.98, staircase=True)
				
				# OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
				self.training_ldcsf = tf.train.GradientDescentOptimizer(self.learning_rate_ldcsf)
				self.training_ldcsf_train = self.training_ldcsf.minimize(self.loss_ldcsf,
					var_list=self.var_list_ldcsf)

			########--------------------------*-------------------------########

			# Add variable initializer.
			self.init_op = tf.global_variables_initializer()

			# create a saver
			self.saver = tf.train.Saver()

			self.merged_summary_sentsim = tf.summary.merge_all('sentsim')
			self.merged_summary_leiods_classification = tf.summary.merge_all('leidos-classification')
			self.merged_summary_ldcsf_classification = tf.summary.merge_all('ldcsf-classification')

	def read_format_sentsim(self,filename_queue):

		# file reader and value generator
		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)
		
		# default format of the file
		record_defaults = [[]] * ((self.max_length*2)+3)
		record_defaults[1].append(0)

		# extract all csv columns
		features = tf.decode_csv(value,record_defaults=record_defaults)

		# make different inputs for two sentences/examples
		example1 = features[:self.max_length]
		example2 = features[self.max_length:self.max_length*2]
		
		example1_len = features[-3]
		example2_len = features[-2]
		# extract label, whether they are similar/1 or not/0
		label = tf.to_float(features[-1])

		label = tf.stack([label])

		return example1, example2, example1_len, example2_len, label

	def input_pipeline_sentsim(self,filenames, batch_size, num_epochs=None):
		filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=num_epochs, shuffle=True)
		example1, example2, example1_len, example2_len, label = self.read_format_sentsim(filename_queue)
		# min_after_dequeue defines how big a buffer we will randomly sample
		#   from -- bigger means better shuffling but slower start up and more
		#   memory used.
		# capacity must be larger than min_after_dequeue and the amount larger
		#   determines the maximum we will prefetch.  Recommendation:
		#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
		min_after_dequeue = 100000
		capacity = min_after_dequeue + 3 * batch_size 
		example1_batch, example2_batch, example1_len_batch, example2_len_batch,\
		label_batch = tf.train.shuffle_batch([example1, example2, example1_len, example2_len, label],
		 	batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue,shapes=None)
		
		return example1_batch, example2_batch, example1_len_batch, example2_len_batch, label_batch


	#		print(document_vector)
	#		print("document vector shape : ",np.array(document_vector).shape)

	#		new_saver = tf.train.import_meta_graph(model_meta_file)
	#		new_saver.restore(sess, tf.train.latest_checkpoint('./'))
'''
class TestClassifier(object):

	def __init__(self,model_path):

	def test_classifier(self):
'''

'''
class TestDocClassifier(object):

	def __init__(self, model):
'''