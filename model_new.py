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
from lib.skipgram import SkipGramGraph
from lib.batch_data_generators import generate_train_batch_data_task_leidos,\
generate_test_batch_data_task_leidos, generate_batch_data_task_ldcsf



class DocClassifier(BaseEstimator, TransformerMixin):

	def __init__(self,embedding_size=100, sent_aggregator=None, task_batch_size=5, valid_size=10,
		learning_rate=0.01, sent_attention_size=100, doc_attention_size=100, sent_embedding_size=100,
		doc_embedding_size=100, sent_lstm_layer=1, doc_lstm_layer=1, leidos_num_classes=50,
		ldcsf_num_classes=12,task_learning_rate=.01, multiatt=True, model_name='test_leidos',
		max_length=50,sentsim_learning_rate=0.01, sentsim_batch_size=20, threshold = 0.5,
		num_threads=20,skipgram_learning_rate=.01, skipgram_batch_size=256, skipgram_num_sampled=64):

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
		self.valid_examples = np.random.choice(500, 16, replace=False)
		self.logs_path = LOGS_PATH+model_name
		self.sentsim_batch_size = sentsim_batch_size
		self.threshold = threshold
		self.skipgram_batch_size = skipgram_batch_size
		self.skipgram_num_sampled = skipgram_num_sampled
		self.skipgram_learning_rate = skipgram_learning_rate

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
		#we unstack and pick only one document to initiate the sentence encoder
#		self.doc_embed_0 = self.doc_embed[0]
#		self.doc_embed_0 = tf.unstack(self.doc_embed)[0]
			
		'''
		we unstack sentlen to get sentence lengths of a single document
		this is because sentence aggregator/aggregator class can only
		take list of sequences and not list of list of sequences
		'''
#		self.sentlen_0 = self.yelp_sentlen_batch[0]
			
		self.sent_aggregator = Aggregator(embedding_size=self.embedding_size, attention_size=self.sent_attention_size,
			n_hidden=self.sent_embedding_size, lstm_layer=self.sent_lstm_layer,idd='sent')

#		self.var_list_sentsim = self.var_list_sentsim + self.sent_aggregator.aggregator_variables
#		self.sent_aggregator.init_attention_aggregator()
#		self.var_list_leidos = self.var_list_leidos + self.sent_aggregator.aggregator_variables

	def _initiate_document_encoder(self):


		self.document_aggregator = DocAggregator(embedding_size=self.embedding_size,
			sent_embedding_size=self.sent_embedding_size, sent_attention_size=self.sent_attention_size,
			doc_attention_size=self.doc_attention_size, doc_embedding_size=self.doc_embedding_size,
			sent_aggregator=self.sent_aggregator, lstm_layer=self.doc_lstm_layer,
			idd='doc', multiatt=self.multiatt)
#		self.document_aggregator.initiate_doc_attention_aggregator(doc_embed=docc_embed,
#			sent_len=self.sent_sentlen_batch, doc_len=self.sent_doclen_batch, num_class=5)

#		self.var_list_sentsim = self.var_list_sentsim + self.document_aggregator.doc_attention_aggregator.aggregator_variables
#		self.var_list_leidos = self.var_list_leidos + self.document_aggregator.doc_attention_aggregator.aggregator_variables
	
	def _yelp_placeholders(self):

		#keep first 15 of labels vector for ldc sf
		#default axis for gather_axis is -1
		self.yelp_doc_batch = tf.placeholder(tf.int32,[None,None,None], name='yelp_document_batch')
		self.yelp_sentlen_batch = tf.placeholder(tf.int32, [None,None], name='yelp_sentlen_batch')
		self.yelp_doclen_batch = tf.placeholder(tf.int32, [None], name='yelp_doclen_batch')
		self.yelp_labels_batch = tf.placeholder(tf.float32, [None,5], name='yelp_labels_batch')

	def _yelp_prediction_layer(self):

		with tf.name_scope('Yelp-prediction-SGD'):
			self.yelp_W = {
			'h1': tf.Variable(tf.random_normal([self.sent_embedding_size, 64])),
			'h2': tf.Variable(tf.random_normal([64, 64])),
			'out': tf.Variable(tf.random_normal([64, 5]))
			}
			self.yelp_b = {
			'b1': tf.Variable(tf.random_normal([64])),
			'b2': tf.Variable(tf.random_normal([64])),
			'out': tf.Variable(tf.random_normal([5]))
			}
			self.var_list_yelp_all.append(self.yelp_W['h1'])
			self.var_list_yelp.append(self.yelp_W['h1'])
			self.var_list_yelp_all.append(self.yelp_W['h2'])
			self.var_list_yelp.append(self.yelp_W['h2'])
			self.var_list_yelp_all.append(self.yelp_W['out'])
			self.var_list_yelp.append(self.yelp_W['out'])

			self.var_list_yelp_all.append(self.yelp_b['b1'])
			self.var_list_yelp.append(self.yelp_b['b1'])
			self.var_list_yelp_all.append(self.yelp_b['b2'])
			self.var_list_yelp.append(self.yelp_b['b2'])
			self.var_list_yelp_all.append(self.yelp_b['out'])
			self.var_list_yelp.append(self.yelp_b['out'])

			# Construct model
			self.yelp_pred = multilayer_perceptron(self.yelp_document_vector, self.yelp_W, self.yelp_b)

			# Define loss and optimizer
			self.yelp_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yelp_pred,
				labels=self.yelp_labels_batch),name='yelp_cost')
			self.yelp_mlp_optimizer_pretrain = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.yelp_cost,
				var_list=self.var_list_yelp, name='yelp_optimizer_pretrain')
			self.yelp_mlp_optimizer_all = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.yelp_cost,
				var_list=self.var_list_yelp_all, name='yelp_optimizer_all')


			self.yelp_acc = tf.equal(tf.argmax(self.yelp_pred, 1), tf.argmax(self.yelp_labels_batch, 1))
			self.yelp_acc = tf.reduce_mean(tf.cast(self.yelp_acc, tf.float32), name='yelp_accuracy')

	def _sentsim_inputpipeline(self):
		##SentSim placeholders
		### training batch extractor
		self.train_sentsimx_batch, self.train_sentsimy_batch, self.train_sentsimx_len_batch,\
		self.train_sentsimy_len_batch,self.train_sentsim_labels_batch =\
		input_pipeline_sentsim(filenames=[DATA_BATCH+'sentsim.csv'],
			batch_size=self.sentsim_batch_size)

		# validation batch extractor
		self.valid_sentsimx_batch, self.valid_sentsimy_batch, self.valid_sentsimx_len_batch,\
		self.valid_sentsimy_len_batch,self.valid_sentsim_labels_batch =\
		input_pipeline_sentsim(filenames=[DATA_BATCH+'valid_es_en.csv'],
			batch_size=250)

	def _sentsim_placeholders(self):

		#### sent encoder parallel sentences
		self.train_sentsimx = tf.placeholder(tf.int32, [None,None], name='sentsim-inputx')
		self.train_sentsimy = tf.placeholder(tf.int32, [None,None], name='sentsim-inputy')
		self.train_sentsimx_len = tf.placeholder(tf.int32, [None], name='sentsim-inputx_len')
		self.train_sentsimy_len = tf.placeholder(tf.int32, [None], name='sentsim-inputy_len')
		self.train_sentsim_labels = tf.placeholder(tf.float32, [None, 1], name='sentsim-outlabel')

	def _get_sentsim_sentence_encodings(self,train_sentsim,train_sentsim_len):

		embed = tf.nn.embedding_lookup(self.embeddings, train_sentsim)
		embed = tf.nn.dropout(embed, self.keep_prob)

		# if using lstm layer
		if self.sent_lstm_layer == 1:
				
			context = self.sent_aggregator.calculate_attention_with_lstm(embed, 
				sequence_length=train_sentsim_len, keep_prob=self.keep_prob)
			context = tf.nn.dropout(context, self.keep_prob)

		# if no lstm layer
		if self.sent_lstm_layer == 0:

			context = self.sent_aggregator.calculate_attention(embed)
			context = tf.nn.dropout(context, self.keep_prob)

		return context

	def _sentsim_prediction_layer(self):

		with tf.name_scope('SentSim-Loss'):
			self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean = loss(self.contextx,
				self.contexty, self.train_sentsim_labels)
			tf.summary.scalar("task_loss_divide", self.cost_mean, collections=['sentsim'])
			tf.summary.scalar("task_loss_match_divide", self.cost_match_mean, collections=['sentsim'])
			tf.summary.scalar("task_loss_mismatch_divide", self.cost_mismatch_mean, collections=['sentsim'])
			
		with tf.name_scope('SentSim-SGD'):
			self.sent_learning_rate = tf.train.exponential_decay(self.sentsim_learning_rate,
				self.global_step, 50000, 0.98, staircase=True)


			self.sentsim_optimizer = tf.train.GradientDescentOptimizer(self.sent_learning_rate)

			self.sentsim_optimizer_train = self.sentsim_optimizer.minimize(self.cost_mean,
				var_list=self.var_list_sentsim)

	def _get_document_vector(self, doc_batch, sentlen_batch, doclen_batch):

		doc_embed = tf.nn.embedding_lookup(self.embeddings, doc_batch)
		doc_embed = tf.nn.dropout(doc_embed, self.keep_prob)
		document_vector = self.document_aggregator.calculate_document_vector(doc_embed,
			sentlen_batch, doclen_batch,keep_prob=self.keep_prob)
		document_vector = tf.nn.dropout(document_vector, self.keep_prob)

		return document_vector
	
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
			self.var_list_sentsim = []
			self.var_list_yelp = []

			#initiate the embeddings for each word in the vocabulary
			self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size,
				self.embedding_size], -1.0, 1.0), name='word_embeddings')
			self.var_list_sentsim.append(self.embeddings)

			################### define all placeholders #####################

			##DocClassifier placeholders
			self.global_step = tf.Variable(0, trainable=False)

			#define yelp placeholders
			self._yelp_placeholders()
			self._sentsim_inputpipeline()
			self._sentsim_placeholders()

			####SF labels
#			self.sf_labels_batch = tf.placeholder(tf.float32, [None,None], name='ldcsf_labels_batch')

			#parameter to control droupput, keep_prob-0.7, means it keeps 0.7 of the data
			self.keep_prob = tf.placeholder("float",name='keep_prob')

			####----initiate sentence encoder and document encoders with attention----#####


			'''
			we initiate the aggregator for encoding sentences
			use lstm_layer=1 for if using lstm layer else 0
			it generates self.sent_aggregator
			'''
			with tf.variable_scope("Doc-AttentionBasedAggregator") as doc_scope:
#			with tf.name_scope('AttentionBasedAggregator') as scope:

				self._initiate_sentence_encoder()

				self._initiate_document_encoder()
				
				self.yelp_document_vector = self._get_document_vector(doc_batch=self.yelp_doc_batch,
					sentlen_batch=self.yelp_sentlen_batch, doclen_batch=self.yelp_doclen_batch)

				doc_scope.reuse_variables()
				self.contextx = self._get_sentsim_sentence_encodings(train_sentsim=self.train_sentsimx,
					train_sentsim_len=self.train_sentsimx_len)
					
				self.contexty = self._get_sentsim_sentence_encodings(train_sentsim=self.train_sentsimy,
					train_sentsim_len=self.train_sentsimy_len)

				'''
				we initiate the aggregator for encoding documents with
				the doc_batch. here also keep lstm_layer, if using lstm
				layer at the document level
				it creates self.doc_aggregator
				'''

			self.var_list_sentsim = self.var_list_sentsim + self.sent_aggregator.aggregator_variables
			self.var_list_yelp = self.var_list_yelp + self.document_aggregator.doc_attention_aggregator.aggregator_variables
			self.var_list_yelp_all = self.var_list_sentsim + self.var_list_yelp


			
			# Store layers weight & bias

			#yelp prediction layer
			self._yelp_prediction_layer()
			self._sentsim_prediction_layer()

			print("sentsim_variables :", self.var_list_sentsim,"\n\n")
			print("yelp_variables :", self.var_list_yelp,"\n\n")
			print("all_variables :", self.var_list_yelp_all,"\n\n")
			# Add variable initializer.
			self.init_op = tf.global_variables_initializer()

			# create a saver
			self.saver = tf.train.Saver(name='model_saver')

			self.merged_summary_skip = tf.summary.merge_all('skip-gram')
			self.merged_summary_sentsim = tf.summary.merge_all('sentsim')
#			self.merged_summary_leiods_classification = tf.summary.merge_all('leidos-classification')
			self.merged_summary_ldcsf_classification = tf.summary.merge_all('ldcsf-classification')

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