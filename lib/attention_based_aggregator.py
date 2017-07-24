#!/usr/bin/python

''' Author : Karan Singla '''

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

#library imports
import numpy as np

#tensorflow imports
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn

from lib.util import _attn_mul_fun, map_fn_mult



def BiRNN(lstm_bw_cell, x, sequence_length=None,idd='sent'):

	'''
	Input Variables
	lstm_bw_cell : 
	x : 
	sequence_length : 
	idd :
	'''

	# Get lstm cell output
	with tf.variable_scope(idd+'lstm1') as vs:
		outputs, states = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32, sequence_length=sequence_length)
		rnn_variables = [v for v in tf.all_variables()
						if v.name.startswith(vs.name)]

	return outputs,rnn_variables

class DocAggregator(object):

	def __init__(self, embedding_size=100, sent_attention_size=None, doc_attention_size=None,doc_embedding_size=None,\
		sent_embedding_size=None,sent_aggregator=None,lstm_layer=1, sent_lstm_layer=1,keep_prob=0.7, idd='doc', multiatt=True):

		self.embedding_size = embedding_size
		self.doc_embedding_size = doc_embedding_size
		self.sent_embedding_size = sent_embedding_size
		self.sent_attention_size = sent_attention_size
		self.doc_attention_size = doc_attention_size
		self.sent_aggregator = sent_aggregator
		self.lstm_layer = lstm_layer
		self.idd = idd
		self.multiatt = multiatt

		self.keep_prob = keep_prob
		self.sent_lstm_layer = sent_lstm_layer

		self.doc_attention_aggregator = Aggregator(embedding_size=self.sent_embedding_size,
				attention_size=self.doc_attention_size, n_hidden=self.doc_embedding_size, 
				lstm_layer=lstm_layer,idd='doc')

		self._initiate_doc_attention_aggregator()

	def _initiate_doc_attention_aggregator(self):

		'''
		doc = tf.placeholder(tf.int32, [None,max_doc_size,max_sent_len], name='doc')
		doc_embed = tf.nn.embedding_lookup(self.embeddings, doc)

		seq_len = tf.placeholder(tf.int32, [None,300], name='seq-len')
		doc_len = tf.placeholder(tf.int32, [None], name='doc-len')

		'''

		#check if sent_aggregator is None or not
		#if None it will initiate a sentence encoder
		if self.sent_aggregator == None:

			print("No sentence aggregator found")
			print("Initiating a sentence aggregator")

			self.sent_aggregator = Aggregator(embedding_size=self.embedding_size, attention_size=self.sent_attention_size,
				n_hidden=self.sent_embedding_size, lstm_layer=self.sent_lstm_layer,idd='sent')

		else:
			print("Using previously initiated sentence aggregator")
	
	def _initiate_sentence_aggregator(self, embed, seq_len):

		with tf.name_scope('AttentionBasedAggregator'):
			self.sent_aggregator = Aggregator(sequence_length=seq_len, embedding_size=self.embedding_size,
				attention_size=self.sent_attention_size, embed = embed, n_hidden=self.sent_embedding_size,\
				lstm_layer=1, keep_prob=0.7,idd='sent')


	def _inititate_doc_aggregator(self, embed, doc_len, doc_attention_size, num_class=None,
		lstm_layer=1, keep_prob=0.7):
		'''
		this is the heler function for initiate_doc_attention_aggregator()
		'''

		with tf.name_scope('DocAttentionBasedAggregator'):

			self.doc_attention_aggregator = Aggregator(sequence_length=doc_len,embedding_size=self.sent_embedding_size,
				attention_size=self.doc_attention_size, embed=embed, n_hidden=self.doc_embedding_size, lstm_layer=lstm_layer,
				keep_prob=keep_prob,idd=self.idd)

#			#if using multiattention framework
#			if self.multiatt == True :
#				self.doc_attention_aggregator.init_multiattention_aggregator_lstm(num_class=num_class)
#			else:
#				self.doc_attention_aggregator.init_attention_aggregator()

#			self.doc_attention_aggregator.init_attention_aggregator()

	def _calculate_sentence_encodings(self, doc_embed,seq_len, keep_prob):


		doc_context = map_fn_mult(self.sent_aggregator.calculate_attention_with_lstm, [doc_embed, seq_len])
#		doc_context = tf.map_fn(self.sent_aggregator.calculate_attention_with_lstm_tuple, (doc_embed,seq_len),
#			dtype=(tf.int32, tf.int32))
#		doc_context = tf.nn.dropout(doc_context, keep_prob)

		return doc_context

	def calculate_document_vector(self,doc_embed, seq_len, doc_len,keep_prob=0.7):

		print("doc_embed :",doc_embed.shape)		
		self.doc_context = self._calculate_sentence_encodings(doc_embed=doc_embed,seq_len=seq_len, keep_prob=keep_prob)
		self.doc_context = tf.nn.dropout(self.doc_context, keep_prob)
#		doc_context = tf.identity(self.doc_context, name='document_sentence_embedding')

		print("DOC_context sentence encodings :",self.doc_context.shape)

		if self.multiatt == True:
			doc_vector = self.doc_attention_aggregator.caluculate_multiattention_with_lstm(self.doc_context, doc_len,
				keep_prob=keep_prob)
			print("DOC_VECTOR :",doc_vector.shape)
		else:
#			doc_vector = math_ops.reduce_mean(self.doc_context, [1])
			doc_vector = self.doc_attention_aggregator.calculate_attention_with_lstm(self.doc_context, doc_len,
				keep_prob=keep_prob)
			print("aggregated DOC_VECTOR :",doc_vector.shape)
#		doc_vector = self.doc_attention_aggregator.calculate_attention_with_lstm(doc_context, doc_len)

#		context_vector = tf.nn.dropout(doc_vector, keep_prob)

		return doc_vector


class Aggregator(object):


	def __init__(self,embedding_size=None, attention_size=None, sequence_length=None, n_hidden=100,
		lstm_layer=1, num_class=1, keep_prob=0.7,idd='sent'):
		
		self.idd = idd #name for this instance
		self.embedding_size = embedding_size #dimension of word-vectors/sentence-vectors 
		self.attention_size = attention_size #dimension of attention vector
		self.keep_prob = keep_prob #droupout keep proability
		self.aggregator_variables = []
		self.num_class = num_class
		self.flag = 0

		if lstm_layer == 1:
			# if lstm_layer == 1, then one  has to also give embed to initiate RNN
			# Define lstm cells with tensorflow
			# Forward direction cell

			self.n_hidden = n_hidden #hidden layer num of features if using lstm_layer=1

			with tf.variable_scope(self.idd+'backward') as vs2:
				self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)


#			with tf.variable_scope(self.idd+'lstm1') as vs:
#				outputs, states = tf.nn.dynamic_rnn(self.lstm_bw_cell, embed, dtype=tf.float32,sequence_length=sequence_length)
#				rnn_variables = [v for v in tf.all_variables()
#								if v.name.startswith(vs.name)]

			print("Initiated Aggregator with LSTM layer")
#			self.aggregator_variables =  self.aggregator_variables + lstm_variables

		else:

			self.n_hidden = embedding_size #hidden layer num of features if using lstm_layer
			print("Initiated Aggregator without LSTM layer")

		self.attention_task = tf.Variable(tf.zeros([self.num_class, self.attention_size]),
			name=self.idd+'attention_vector')
		self.trans_weights = tf.Variable(tf.random_uniform([self.n_hidden, self.attention_size], -1.0, 1.0),
			name=self.idd+'transformation_weights')
		self.trans_bias = tf.Variable(tf.zeros([self.attention_size]), name=self.idd+'_trans_bias')

		self.aggregator_variables.append(self.attention_task)
		self.aggregator_variables.append(self.trans_weights)
		self.aggregator_variables.append(self.trans_bias)

			# Backward direction cell
#			with tf.variable_scope('backward'):
#				self.lstm_fw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

	def average(self, embed):

		context_vector = math_ops.reduce_mean(embed, [1])
#		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def average_with_lstm(self, embed):

		'''
		for using this method make sure
		that lstm_layer == 1 while initiating the 
		aggregator
		'''

		# get BiRNN outputs
		outputs = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)
#		outputs = tf.nn.dropout(outputs, self.keep_prob)


		context_vector = math_ops.reduce_mean(outputs, [1])
#		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def calculate_multiattention(self,embed):

		embeddings_flat = tf.reshape(embed, [-1, self.n_hidden])

		# tanh transformation of embeddings
		keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
			self.trans_weights), self.trans_bias))

		# reshape the keys according to our embed vector
		keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))

		context_vectors = []

		for i in range(0,len(self.attention_task)):

			

			# Now calculate the attention-weight vector.

			# tanh transformation of embeddings
			keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
				self.trans_weights), self.trans_bias))

			# reshape the keys according to our embed vector
			keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))

			# calculate score for each word embedding and take softmax on it
			scores = math_ops.reduce_sum(keys * self.attention_task[i], [2])
			alignments = nn_ops.softmax(scores)

			# expand aligments dimension so that we can multiply it with embed tensor
			alignments = array_ops.expand_dims(alignments,2)

			# generate context vector by making 
			context_vector = math_ops.reduce_sum(alignments * embed, [1])
#			context_vector = tf.nn.dropout(context_vector, self.keep_prob)
			context_vectors.append(context_vector)

		context_vectors = tf.stack(context_vectors)
		return context_vectors

	def caluculate_multiattention_with_lstm(self,embed, sequence_length, keep_prob):

		context_vectors = []

		# get BiRNN outputs
		outputs,rnn_variables = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)

		if self.flag == 0:

			self.aggregator_variables = self.aggregator_variables + rnn_variables
			self.flag = self.flag + 1		
#		outputs = tf.nn.dropout(outputs, keep_prob)

		context_vectors = self.calculate_multiattention(outputs)
		return context_vectors

	def calculate_attention(self, embed):

		embeddings_flat = tf.reshape(embed, [-1, self.n_hidden])

		# Now calculate the attention-weight vector.

		# tanh transformation of embeddings
		keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
			self.trans_weights), self.trans_bias))

		# reshape the keys according to our embed vector
		keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(embed)[:-1], [self.attention_size]]))

		# calculate score for each word embedding and take softmax on it
		scores = math_ops.reduce_sum(keys * self.attention_task, [2])
		alignments = nn_ops.softmax(scores)

		# expand aligments dimension so that we can multiply it with embed tensor
		alignments = array_ops.expand_dims(alignments,2)

		# generate context vector by making 
		context_vector = math_ops.reduce_sum(alignments * embed, [1])
#		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def calculate_attention_with_lstm(self, embed, sequence_length, keep_prob=1.0):

		'''
		this method only works if you use
		init_attention_aggregator_lstm
		sequence_length : 1D matrix having original length of sentences of inputs in x
		'''
		print(embed.shape)

		# get BiRNN outputs
		outputs,rnn_variables = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)

		if self.flag == 0:

			self.aggregator_variables = self.aggregator_variables + rnn_variables
			self.flag = self.flag + 1
#		outputs = tf.nn.dropout(outputs, keep_prob)

		context_vector = self.calculate_attention(outputs)
#		context_vector = tf.nn.dropout(context_vector, keep_prob)
		return context_vector

	def calculate_attention_with_lstm_doc(self, embed, sequence_length, doc_len, keep_prob=1.0):

		'''
		this method only works if you use
		init_attention_aggregator_lstm
		sequence_length : 1D matrix having original length of sentences of inputs in x
		'''
		print(embed.shape)
		embed = tf.slice(embed,[0,0,0],[doc_len,50,200])
		if self.attention_task is None:
			print("Initiating attention mechanism with lstm")
			init_attention_aggregator_lstm()

		# get BiRNN outputs
		outputs = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)
#		outputs = tf.nn.dropout(outputs, keep_prob)

		embeddings_flat = tf.reshape(outputs, [-1, self.n_hidden])

		# tanh transformation of embeddings
		keys_flat = tf.tanh(tf.add(tf.matmul(embeddings_flat,
			self.trans_weights), self.trans_bias))

		# reshape the keys according to our embed vector
		keys = tf.reshape(keys_flat, tf.concat(axis=0,values=[tf.shape(outputs)[:-1], [self.attention_size]]))
		
		# calculate score for each word embedding and take softmax on it
		scores = _attn_mul_fun(keys, self.attention_task)
		alignments = nn_ops.softmax(scores)

		# expand aligments dimension so that we can multiply it with embed tensor
		alignments = array_ops.expand_dims(alignments,2)

		# generate context vector by making 
		context_vector = math_ops.reduce_sum(alignments * outputs, [1])
#		context_vector = tf.nn.dropout(context_vector, keep_prob)
		return context_vector
