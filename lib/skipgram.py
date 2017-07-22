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

#library imports
import numpy as np

#tensorflow imports
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn


class SkipGramGraph(object):

	def __init__(self,embed,train_skip_labels,vocabulary_size,embedding_size,num_sampled,
		skipgram_learning_rate,global_step,name='Word2Vec'):
		# Construct the variables for the NCE loss
		self.nce_weights = tf.Variable(
			tf.truncated_normal([vocabulary_size, embedding_size],
				stddev=1.0 / math.sqrt(embedding_size)), name=name+'-skipgram-nce-weights')

		self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]), name=name+'-skipgram-nce-biases')

		with tf.name_scope('Skipgram-NCE-Loss'):

			self.skip_loss = tf.reduce_mean(
				tf.nn.nce_loss(weights=self.nce_weights,
					biases=self.nce_biases,
					labels=train_skip_labels,
					inputs=embed,
					num_sampled=num_sampled,
					num_classes=vocabulary_size))

		with tf.name_scope('Skip-gram-SGD'):
			self.skipgram_learning_rate = tf.train.exponential_decay(skipgram_learning_rate, global_step,
                                           50000, 0.98, staircase=True)
		

			self.skip_optimizer = tf.train.GradientDescentOptimizer(self.skipgram_learning_rate).minimize(self.skip_loss,
				global_step=global_step)

	def input_pipeline_skipgram(self,filenames, batch_size, num_epochs=None):
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
