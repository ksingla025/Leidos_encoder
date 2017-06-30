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

from lib.util import _attn_mul_fun

def BiRNN(lstm_bw_cell, x, sequence_length,idd='sent'):

	'''
	Input Variables
	lstm_bw_cell : 
	x : 
	sequence_length : 
	idd :
	'''

	# Get lstm cell output
	with tf.variable_scope(idd+'lstm1', reuse=True):
		outputs, states = tf.nn.dynamic_rnn(lstm_bw_cell, x, dtype=tf.float32, sequence_length=sequence_length)

	return outputs

class DocAggregator(object):

	def __init__(self, embedding_size=100, sent_attention_size=None, doc_attention_size=None,doc_embedding_size=None,\
		sent_embedding_size=None,sent_aggregator=None,lstm_layer=1, keep_prob=0.7, idd='doc'):

		self.embedding_size = embedding_size
		self.doc_embedding_size = doc_embedding_size
		self.sent_embedding_size = sent_embedding_size
		self.sent_attention_size = sent_attention_size
		self.doc_attention_size = doc_attention_size
		self.sent_aggregator = sent_aggregator
		self.lstm_layer = lstm_layer
		self.idd = idd

		self.keep_prob = keep_prob

	def initiate_doc_attention_aggregator(self, doc_emded, seq_len, doc_len):

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

			#get 1 document from the batch
			embed = tf.unstack(doc_embed)[0]

			#get sentence length of 1 document from the batch
			seq_len_unstack = tf.unstack(seq_len)[0]

			_inititate_sentence_aggregator(emded,seq_len)

		doc_context = _calculate_sentence_encodings(doc_embed,seq_len)

		_inititate_doc_aggregator(emded=doc_context, doc_len=doc_len,lstm_layer=1, keep_prob=self.keep_prob)

	
	def _initiate_sentence_aggregator(embed,seq_len):

		with tf.name_scope('AttentionBasedAggregator'):
			self.sent_aggregator = Aggregator(sequence_length=seq_len,embedding_size=self.embedding_size,
				attention_size=self.sent_attention_size, embed = embed, n_hidden=self.sent_embedding_size,\
				lstm_layer=1, keep_prob=0.7,idd='sent')

			if self.lstm_layer == 1:
				self.sent_aggregator.init_attention_aggregator_lstm()
			else:
				self.sent_aggregator.init_attention_aggregator()

	def _inititate_doc_aggregator(emded, doc_len, doc_attention_size, lstm_layer=1,keep_prob=0.7):
		'''
		this is the heler function for initiate_doc_attention_aggregator()
		'''

		with tf.name_scope('DocAttentionBasedAggregator'):

			self.doc_attention_aggregator = Aggregator(sequence_length=doc_len,embedding_size=self.sent_embedding_size,
				attention_size=self.doc_attention_size, embed=embed, n_hidden=self.doc_embedding_size, lstm_layer=lstm_layer, keep_prob=0.7,idd=self.idd)

			if self.lstm_layer == 1:
				self.doc_attention_aggregator.init_attention_aggregator_lstm()
			else:
				self.doc_attention_aggregator.init_attention_aggregator()

	def _calculate_sentence_encodings(doc_embed,seq_len):

		doc_unstack = tf.unstack(doc_embed)

		# document lengths of each sentence in each batch for doc
		seq_len_unstack = tf.unstack(seq_len)

		# initialize aggregator		
		
		doc_context = []

		for i in range(0,len(docunstack)):

			sequence_length = tf.reshape(seq_len_unstack[i],[-1])

			context = self.sent_aggregator.calculate_attention_with_lstm(docunstack[i], sequence_length)
			context = tf.nn.dropout(context, self.keep_prob)

			doc_context.append(context)

		#stack all (doc_batch_size) doc1 context vectors
		doc_context = tf.stack(self.doc_context)

		return doc_context

	def calculate_document_vector(self,doc_embed, seq_len, doc_len,keep_prob=0.7):

		doc_context = _calculate_sentence_encodings(doc_embed,seq_len)

		doc_vector = self.sent_aggregator.calculate_attention_with_lstm(doc_context, doc_len)

		context_vector = tf.nn.dropout(doc_vector, keep_prob)



class Aggregator(object):


	def __init__(self,embedding_size=None, attention_size=None, embed = None, sequence_length=None, n_hidden=100, lstm_layer=1, keep_prob=0.7,idd='sent'):
		
		self.idd = idd #name for this instance
		self.embedding_size = embedding_size #dimension of word-vectors/sentence-vectors 
		self.attention_size = attention_size #dimension of attention vector
		self.n_hidden = n_hidden #hidden layer num of features if using lstm_layer=1
		self.keep_prob = keep_prob #droupout keep proability

		if lstm_layer == 1:
			# if lstm_layer == 1, then one  has to also give embed to initiate RNN
			# Define lstm cells with tensorflow
			# Forward direction cell

			with tf.variable_scope(self.idd+'backward'):
				self.lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)

			with tf.variable_scope(self.idd+'lstm1'):
				outputs, states = tf.nn.dynamic_rnn(self.lstm_bw_cell, embed, dtype=tf.float32,sequence_length=sequence_length)

			print("Initiated Aggregator with LSTM layer")

		else:
			print("Initiated Aggregator without LSTM layer")

			# Backward direction cell
#			with tf.variable_scope('backward'):
#				self.lstm_fw_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

#	def multiattention_based_aggregator(self, embed, att_type=50):



	def init_attention_aggregator(self):

		# make the embeddings flat [batch_size*sen_length*embedding_size,1]
		
		self.attention_task = tf.Variable(tf.random_uniform([1, self.attention_size], -1.0, 1.0),
			name=self.idd+'attention_vector')

		self.trans_weights = tf.Variable(tf.zeros([self.embedding_size, self.attention_size]),
			name=self.idd+'transformation_weights')

		self.trans_bias = tf.Variable(tf.zeros([self.attention_size]), name=self.idd+'_trans_bias')

	def init_attention_aggregator_lstm(self):

		self.attention_task = tf.Variable(tf.random_uniform([1, self.attention_size], -1.0, 1.0),
			name=self.idd+'attention_vector')

		self.trans_weights = tf.Variable(tf.zeros([self.n_hidden, self.attention_size]),
		name=self.idd+'transformation_weights')

		self.trans_bias = tf.Variable(tf.zeros([self.attention_size]), name=self.idd+'_trans_bias')

#	def init_multiattention_aggregator(self, label_type=50):
#	def init_multiattention_aggregator_lstm(self, label_type=50):

	def average(self, embed):

		context_vector = math_ops.reduce_mean(embed, [1])
		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def average_with_lstm(self, embed):

		'''
		for using this method make sure
		that lstm_layer == 1 while initiating the 
		aggregator
		'''

		# get BiRNN outputs
		outputs = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)
		outputs = tf.nn.dropout(outputs, self.keep_prob)


		context_vector = math_ops.reduce_mean(outputs, [1])
		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def calculate_attention(self, embed):

		if self.attention_task is None:
			print("Initiating attention mechanism")
			init_attention_aggregator()

		embeddings_flat = tf.reshape(embed, [-1, self.embedding_size])

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
		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector

	def calculate_attention_with_lstm(self, embed, sequence_length):

		'''
		this method only works if you use
		init_attention_aggregator_lstm
		sequence_length : 1D matrix having original length of sentences of inputs in x
		'''
		if self.attention_task is None:
			print("Initiating attention mechanism with lstm")
			init_attention_aggregator_lstm()

		# get BiRNN outputs
		outputs = BiRNN(self.lstm_bw_cell, embed, sequence_length,idd=self.idd)
		outputs = tf.nn.dropout(outputs, self.keep_prob)

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
		context_vector = tf.nn.dropout(context_vector, self.keep_prob)
		return context_vector
