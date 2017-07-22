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
#		self.sent_aggregator.init_attention_aggregator()

#		self.var_list_sentsim = self.var_list_sentsim + self.sent_aggregator.aggregator_variables
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

			#initiate the embeddings for each word in the vocabulary
			self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size,
				self.embedding_size], -1.0, 1.0), name='word_embeddings')

			################### define all placeholders #####################

			##DocClassifier placeholders
			self.global_step = tf.Variable(0, trainable=False)

			#keep first 15 of labels vector for ldc sf
			#default axis for gather_axis is -1
			self.sent_doc_batch = tf.placeholder(tf.int32,[None,None,None], name='document_batch')
			self.yelp_doc_batch = tf.placeholder(tf.int32,[None,None,None], name='yelp_document_batch')
			self.yelp_sentlen_batch = tf.placeholder(tf.int32, [None,None], name='yelp_sentlen_batch')
			self.yelp_doclen_batch = tf.placeholder(tf.int32, [None], name='yelp_doclen_batch')
			self.yelp_labels_batch = tf.placeholder(tf.float32, [None,5], name='yelp_labels_batch')


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
			with tf.name_scope('AttentionBasedAggregator') as scope:
				self._initiate_sentence_encoder()

				'''
				we initiate the aggregator for encoding documents with
				the doc_batch. here also keep lstm_layer, if using lstm
				layer at the document level
				it creates self.doc_aggregator
				'''
				self._initiate_document_encoder()


			self.yelp_doc_embed = tf.nn.embedding_lookup(self.embeddings, self.yelp_doc_batch)
			self.yelp_doc_embed = tf.nn.dropout(self.yelp_doc_embed, self.keep_prob)
			self.yelp_document_vector = self.document_aggregator.calculate_document_vector(self.yelp_doc_embed,
			self.yelp_sentlen_batch, self.yelp_doclen_batch,keep_prob=self.keep_prob)


			# Store layers weight & bias
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

			# Construct model
			self.yelp_pred = multilayer_perceptron(self.yelp_document_vector, self.yelp_W, self.yelp_b)

			# Define loss and optimizer
			self.yelp_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yelp_pred,
				labels=self.yelp_labels_batch),name='yelp_cost')
			self.yelp_mlp_optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.yelp_cost,name='yelp_optimizer')

			self.yelp_acc = tf.equal(tf.argmax(self.yelp_pred, 1), tf.argmax(self.yelp_labels_batch, 1))
			self.yelp_acc = tf.reduce_mean(tf.cast(self.yelp_acc, tf.float32), name='yelp_accuracy')

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