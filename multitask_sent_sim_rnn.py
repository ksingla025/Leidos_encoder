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

############### Utility Functions ####################
class DocClassifier(BaseEstimator, TransformerMixin):

	def __init__(self,embedding_size=200, task_batch_size=5, valid_size=10,
		learning_rate=0.01, num_steps=5000000, sent_attention_size=150,
		doc_attention_size=150, sent_embedding_size=150, doc_embedding_size=150,
		lstm_layer=1,keep_prob=0.7,num_classes=50):

		#set parameters
		self.embedding_size = embedding_size
		self.task_batch_size = task_batch_size
		self.valid_size = valid_size
		self.learning_rate = .01
		self.num_steps = num_steps
		self.sent_attention_size = sent_attention_size
		self.doc_attention_size = doc_attention_size
		self.sent_embedding_size = sent_embedding_size
		self.doc_embedding_size = doc_embedding_size
		self.lstm_layer = lstm_layer
		self.keep_prob = keep_prob
		self.num_classes = num_classes

		self.doc_classifier_graph()

	def doc_classifier_graph(self):
		'''
		this is the main network function for document classifier
		Features ( can use ):
		1. LSTM (with attention) based encoding of sentences
		2. Pre-load a sentence encoder
		3. LSTM (with attention) based encoding of documents from sentences
		4. Classifier
		'''
		self.doc_batch = tf.placeholder(tf.int32, [None,None,None], name='document_batch')
		self.sentlen_batch = tf.placeholder(tf.int32, [None,None], name='sentlen_batch')
		self.labels_batch = tf.placeholder(tf.int32, [None,self.num_classes], name='labels_batch')

		with tf.name_scope('Doc_AttentionBasedAggregator'):

			document_aggregator = DocAggregator(embedding_size=self.embedding_size, sent_embedding_size=self.sent_embedding_size,\
				sent_attention_size=self.sent_attention_size, doc_attention_size=self.doc_attention_size,\
				doc_embedding_size=self.doc_embedding_size, sent_aggregator=None, lstm_layer=1, idd='doc')








class MultiTask(BaseEstimator, TransformerMixin):

	def __init__(self, vocabulary_size=500000, embedding_size=200, batch_size=256,
		multi_batch_size=5, task_batch_size=32, skip_window=5, skip_multi_window = 5,
		num_sampled=64, min_count = 5, valid_size=16, valid_window=500, 
		skip_gram_learning_rate=0.01, sen_length=20, sentsim_learning_rate=0.01,
		num_steps=1400001, task_mlp_start=0, task_mlp_hidden=50, 
		attention='true', n_hidden=100, attention_size = 150, joint='true', 
		name='test', max_length=50, lstm_layer=1, keep_prob = 0.7,
		num_threads=10,num_classes=2, loss_margin=0.0):

		# set parameters
		self.vocabulary_size = vocabulary_size # size of vocabulary
		self.embedding_size = embedding_size # Dimension of the embedding vectorself.
		self.batch_size = batch_size # mono-lingual batch size
		self.multi_batch_size = multi_batch_size # multi-lingual batch size
		self.task_batch_size = task_batch_size # task batch size
		self.skip_window = skip_window # skip window for mono-skip gram batch
		self.skip_multi_window = skip_multi_window # window for soft-alignment
		self.sen_length = sen_length # upper bar on task input sentence
		self.num_sampled = num_sampled # Number of negative examples to sample.
		self.valid_size = valid_size    # Random set of words to evaluate similarity on.
		self.valid_window = valid_window  # Only pick dev samples in the head of the distribution.
		self.valid_examples = np.random.choice(self.valid_window, self.valid_size, replace=False)
		self.attention = attention # attention "true"/"false"
		self.lstm_layer = lstm_layer # method of attention bilstm / direct
		self.attention_size = attention_size
		self.joint = joint # joint training or not "true"/"false"
		self.num_steps = num_steps # total number of steps
		self.task_mlp_start = task_mlp_start # step to start task 1 : keep low for joint = "true"
		self.logs_path = LOGS_PATH + name # path to log file for tensorboard
		self.model_path = MODEL_PATH + name
		self.num_threads = num_threads # number of threads to use
		self.task_mlp_hidden = task_mlp_hidden # neurons in hidden layer for prediction
		self.skip_gram_learning_rate = skip_gram_learning_rate # skip-gram learning rate
		self.min_count = min_count # minimum count of each word
		#task_mlp parameters
		self.sentsim_learning_rate = sentsim_learning_rate
		self.num_classes = num_classes
		self.n_hidden = n_hidden # hiddent units for LSTM cell
		self.max_length = max_length
		self.loss_margin = loss_margin

		# initiate graph
		self.graph = tf.Graph()


#		self._init_graph()
		

		print("Class & Graph Initialized")

	def _build_dictionaries(self):

		print("Loading Data Files")
		
		self.dictionary = cPickle.load(open(DATA_ID+"dictionary.p", 'rb'))
		self.reverse_dictionary = cPickle.load(open(DATA_ID+"reverse_dictionary.p", 'rb'))
		print("dictionaries loaded")

		self.vocabulary_size = len(self.dictionary.keys())

		

	def sentsim_task_graph(self):

		# training batch extractor
		self.train_sentsimx_batch, self.train_sentsimy_batch, self.train_sentsimx_len_batch, self.train_sentsimy_len_batch,\
		self.train_sentsim_labels_batch = self.input_pipeline_sentsim(filenames=[DATA_BATCH+'sentsim.csv'],
			batch_size=self.task_batch_size)

		# validation batch extractor
		self.valid_sentsimx_batch, self.valid_sentsimy_batch, self.valid_sentsimx_len_batch, self.valid_sentsimy_len_batch,\
		self.valid_sentsim_labels_batch = self.input_pipeline_sentsim(filenames=[DATA_BATCH+'valid_es_en.csv'],
			batch_size=self.task_batch_size)

		self.train_sentsimx = tf.placeholder(tf.int32, [None,None], name='sentsim-inputx')
		self.train_sentsimy = tf.placeholder(tf.int32, [None,None], name='sentsim-inputy')
		self.train_sentsimx_len = tf.placeholder(tf.int32, [None], name='sentsim-inputx_len')
		self.train_sentsimy_len = tf.placeholder(tf.int32, [None], name='sentsim-inputy_len')
		self.train_sentsim_labels = tf.placeholder(tf.float32, [None, 1], name='sentsim-outlabel')

		self.keep_prob = tf.placeholder("float")

		#get embeddings for x and y input sentence
		self.embedx = tf.nn.embedding_lookup(self.embeddings, self.train_sentsimx)
		self.embedx = tf.nn.dropout(self.embedx, self.keep_prob)

		self.embedy = tf.nn.embedding_lookup(self.embeddings, self.train_sentsimy)
		self.embedy = tf.nn.dropout(self.embedy, self.keep_prob)

		#use attention aggregator is using attention
		if self.attention == 'true':

			# initialize attention aggregator
			with tf.name_scope('AttentionBasedAggregator'):
				self.attention_aggregator = Aggregator(sequence_length=self.train_sentsimx_len,embedding_size=self.embedding_size,
					attention_size=self.attention_size, embed = self.embedx, n_hidden=100, lstm_layer=1, keep_prob=0.7,idd='sent')

				if self.lstm_layer == 1:
					self.attention_aggregator.init_attention_aggregator_lstm()
				else:
					self.attention_aggregator.init_attention_aggregator()					

			# if using lstm layer
			if self.lstm_layer == 1:
				
				self.contextx = self.attention_aggregator.calculate_attention_with_lstm(self.embedx,self.train_sentsimx_len)
				self.contextx = tf.nn.dropout(self.contextx, self.keep_prob)

				self.contexty = self.attention_aggregator.calculate_attention_with_lstm(self.embedx,self.train_sentsimx_len)
				self.contexty = tf.nn.dropout(self.contexty, self.keep_prob)

			# if no lstm layer
			if self.lstm_layer == 0:

				self.contextx = self.attention_aggregator.calculate_attention(self.embedx)
				self.contextx = tf.nn.dropout(self.contextx, self.keep_prob)

				self.contexty = self.attention_aggregator.calculate_attention(self.embedy)
				self.contexty = tf.nn.dropout(self.contexty, self.keep_prob)



		with tf.name_scope('Task-Loss'):
			self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean = loss(self.contextx, self.contexty, self.train_sentsim_labels)
#			self.cost = loss(self.contextx, self.contexty, self.train_sentsim_labels,self.loss_margin)
    		# Minimize error using cross entropy
#			self.cost = tf.reduce_mean(-tf.reduce_sum(self.train_sentsim_labels*tf.log(self.pred), axis=1))
		with tf.name_scope('Task-SGD'):
			self.learning_rate = tf.train.exponential_decay(self.sentsim_learning_rate, self.global_step,
				50000, 0.98, staircase=True)


			self.task_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_mean,
				global_step=self.global_step)

		tf.summary.scalar("task_loss_divide", self.cost_mean, collections=['polarity-task'])
		tf.summary.scalar("task_loss_match_divide", self.cost_match_mean, collections=['polarity-task'])
		tf.summary.scalar("task_loss_mismatch_divide", self.cost_mismatch_mean, collections=['polarity-task'])


	def _init_graph(self):

		'''
		Define Graph
		'''

		with self.graph.as_default(), tf.device('/cpu:0'):
			
			# shared embedding layer
			self.embeddings = tf.Variable(
				tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0),
				name='embeddings')

			#training batch extractor
			self.train_skip_inputs, self.train_skip_labels = self.input_pipeline(filenames=[DATA_BATCH+"mono.csv",
					DATA_BATCH+"bi_train.csv"], batch_size=self.batch_size)

#			self.train_skip_inputs = tf.placeholder(tf.int32, name='skip-gram-input')
#			self.train_skip_labels = tf.placeholder(tf.int32, name='skip-gram-output')
			
			self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32, name = 'valid-dataset')

			# step to mamnage decay
			self.global_step = tf.Variable(0, trainable=False)

			# Look up embeddings for skip inputs.
			self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_skip_inputs)

			# Construct the variables for the NCE loss
			self.nce_weights = tf.Variable(
				tf.truncated_normal([self.vocabulary_size, self.embedding_size],
					stddev=1.0 / math.sqrt(self.embedding_size)), name='nce_weights')

			self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]), name='nce_biases')

			with tf.name_scope('Skip-gram-NCE-Loss'):
				self.skip_loss = tf.reduce_mean(
					tf.nn.nce_loss(weights=self.nce_weights,
						biases=self.nce_biases,
						labels=self.train_skip_labels,
						inputs=self.embed,
						num_sampled=self.num_sampled,
						num_classes=self.vocabulary_size))

			with tf.name_scope('Skip-gram-SGD'):
				self.learning_rate = tf.train.exponential_decay(self.skip_gram_learning_rate, self.global_step,
                                           50000, 0.98, staircase=True)
		

				self.skip_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.skip_loss,
				 global_step=self.global_step)
		#		self.skip_optimizer = tf.train.GradientDescentOptimizer(self.skip_gram_learning_rate).minimize(self.skip_loss)

			# Create a summary to monitor cost tensor
			tf.summary.scalar("skip_loss", self.skip_loss, collections=['skip-gram'])

			#------------------------ task_mlp Loss and Optimizer ---------------------
			with tf.name_scope('Sentsim-graph'):
				self.sentsim_task_graph()


			# Compute the cosine similarity between minibatch examples and all embeddings.
			self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1,
				keep_dims=True))
			self.normalized_embeddings = self.embeddings / self.norm

			self.valid_embeddings = tf.nn.embedding_lookup(
				self.normalized_embeddings, self.valid_dataset)
			self.similarity = tf.matmul(self.valid_embeddings, self.normalized_embeddings,
				transpose_b=True)


			# Add variable initializer.
			self.init_op = tf.global_variables_initializer()
			
			# create a saver
			self.saver = tf.train.Saver()

			self.merged_summary_skip = tf.summary.merge_all('skip-gram')
			self.merged_summary_task_mlp = tf.summary.merge_all('polarity-task')
	#		self.merged_summary_task_mlp_valid = tf.summary.merge_all('task_mlp-valid')

	def read_format_skipgram(self,filename_queue):

		# file reader and value generator
		reader = tf.TextLineReader()
		key, value = reader.read(filename_queue)

		# default format of the file
		record_defaults = [[1], [1]]

		# get values in csv files
		col1, col2 = tf.decode_csv(value,record_defaults=record_defaults)
#		
		# col1 is input, col2 is predicted label
		label = tf.stack([col2])
		return col1, label

	def input_pipeline(self,filenames, batch_size, num_epochs=None):
		filename_queue = tf.train.string_input_producer(
			filenames, num_epochs=num_epochs, shuffle=True)
		example, label = self.read_format_skipgram(filename_queue)
		# min_after_dequeue defines how big a buffer we will randomly sample
		#   from -- bigger means better shuffling but slower start up and more
		#   memory used.
		# capacity must be larger than min_after_dequeue and the amount larger
		#   determines the maximum we will prefetch.  Recommendation:
		#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
		min_after_dequeue = 50000
		capacity = min_after_dequeue + 3 * batch_size
		example_batch, label_batch = tf.train.shuffle_batch(
			[example, label], batch_size=batch_size, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
		return example_batch, label_batch

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

	def fit(self):

		self._build_dictionaries()

		self._init_graph()

		# create a session
		coord = tf.train.Coordinator()

		self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(
			intra_op_parallelism_threads=self.num_threads))

		# with self.sess as session:
		session = self.sess

		threads = tf.train.start_queue_runners(sess=session, coord=coord)

		session.run(self.init_op)

		average_loss = 0
		sentsim_average_loss = 0
		cost_init = 99999.0 #initialize the cost to a high value

		# op to write logs to Tensorboard
		summary_writer = tf.summary.FileWriter(self.logs_path, graph=self.graph)

		print("Initialized")

		for step in range(self.num_steps):

			print(step)
			#if we are doing joint training
			if self.joint == 'true':

				#run session

				# skip-gram step
				
				_, loss_val,summary = session.run([self.skip_optimizer, self.skip_loss,
					self.merged_summary_skip])
				# create batches for sentence similarity task
				train_sentsimx_batch, train_sentsimy_batch, train_sentsimx_len_batch, train_sentsimy_len_batch,\
				train_sentsim_labels_batch = session.run([self.train_sentsimx_batch,
					self.train_sentsimy_batch, self.train_sentsimx_len_batch, self.train_sentsimy_len_batch,
					self.train_sentsim_labels_batch])
				
				
				# run the sentence similarity task
				embedx,embedy = session.run([self.embedx, self.embedy],
					feed_dict={self.train_sentsimx: train_sentsimx_batch, self.train_sentsimy: train_sentsimy_batch,
					self.train_sentsim_labels: train_sentsim_labels_batch, self.keep_prob : 0.7 })

#				print embedx.shape
#				print embedy.shape

				contextx,contexty = session.run([self.contextx, self.contexty],
					feed_dict={self.train_sentsimx: train_sentsimx_batch, self.train_sentsimy: train_sentsimy_batch,
					self.train_sentsimx_len: train_sentsimx_len_batch, self.train_sentsimy_len: train_sentsimy_len_batch,
					self.train_sentsim_labels: train_sentsim_labels_batch, self.keep_prob : 0.7 })

#				print contextx.shape
#				print contexty.shape
				_, sentsim_loss, summary_sentsim = session.run([self.task_optimizer, self.cost_mean,self.merged_summary_task_mlp],
					feed_dict={self.train_sentsimx: train_sentsimx_batch, self.train_sentsimy: train_sentsimy_batch,
					self.train_sentsimx_len: train_sentsimx_len_batch, self.train_sentsimy_len: train_sentsimy_len_batch,
					self.train_sentsim_labels: train_sentsim_labels_batch, self.keep_prob : 0.7 })

				#add  loss summary at step
				summary_writer.add_summary(summary, step)
				summary_writer.add_summary(summary_sentsim, step)

				#add loss 
				average_loss += loss_val
				sentsim_average_loss += sentsim_loss
			
			if step % 1000 == 0:

				# read the validation data batch by batch and compute total accuracy
				total_valid_accuracy = 0

				# create batches for sentence similarity task
				valid_sentsimx_batch, valid_sentsimy_batch, valid_sentsimx_len_batch, valid_sentsimy_len_batch,\
				valid_sentsim_labels_batch = session.run([self.valid_sentsimx_batch, self.valid_sentsimy_batch,
					self.valid_sentsimx_len_batch, self.valid_sentsimy_len_batch, self.valid_sentsim_labels_batch])


				cost_mean, cost_match_mean, cost_mismatch_mean = session.run([self.cost_mean, self.cost_match_mean, self.cost_mismatch_mean],
					feed_dict={self.train_sentsimx: valid_sentsimx_batch, self.train_sentsimy: valid_sentsimy_batch,
					self.train_sentsimx_len: valid_sentsimx_len_batch, self.train_sentsimy_len: valid_sentsimy_len_batch,
					self.train_sentsim_labels: valid_sentsim_labels_batch, self.keep_prob : 0.7 })
				#valid_accuracy = self.acc.eval({self.train_sentsimx: valid_sentsimx_batch, 
				#		self.train_sentsimy: valid_sentsimy_batch, self.train_sentsim_labels: valid_sentsim_labels_batch,
				#		self.keep_prob: 1.0}, session=session)

				#if cost_recuces then save the model to this new checkpoint
				if cost_mean < cost_init:
					cost_init = cost_mean
					self.saver.save(session,self.model_path)

				summary = tf.Summary(value=[tf.Summary.Value(tag="valid-loss",
					 simple_value=float(cost_mean))])

				summary_match = tf.Summary(value=[tf.Summary.Value(tag="valid-match-loss",
					 simple_value=float(cost_match_mean))])

				summary_mismatch = tf.Summary(value=[tf.Summary.Value(tag="valid-mismatch-loss",
					 simple_value=float(cost_mismatch_mean))])
					
				summary_writer.add_summary(summary, step)
				summary_writer.add_summary(summary_match, step)
				summary_writer.add_summary(summary_mismatch, step)
			

			if step % 500 == 0:
				if step > 0:
					average_loss /= 500
					sentsim_average_loss /= 500
#					task_mlp_average_loss /= 2000
				# The average loss is an estimate of the loss over the last 2000 batches.
				print("Average loss of skip-gram step ", step, ": ", average_loss)
				print("Average loss of sent-sim at step ", step, ": ", sentsim_average_loss)
				average_loss = 0
				sentsim_average_loss = 0

			
			if step % 10000 == 0:
				sim = self.similarity.eval(session=session)
				for i in xrange(self.valid_size):
					valid_word = self.reverse_dictionary[self.valid_examples[i]]
					top_k = 8  # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k + 1]
					log_str = "Nearest to %s:" % valid_word
					for k in xrange(top_k):
						close_word = self.reverse_dictionary[nearest[k]]
						log_str = "%s %s," % (log_str, close_word)
					print(log_str)
			
		final_embeddings = session.run(self.normalized_embeddings)
		self.final_embeddings = final_embeddings
		return self
