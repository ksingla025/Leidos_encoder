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
from lib.ldc_util import *
from lib.attention_based_aggregator import *


class ClassifierPredict(object):

	def __init__(self,model='./models/test3',parameters='./lib/parameters.json',
		theme_dic = DATA_ID+'ldcsf_theme_dic.p'):

		"""Step 0: load trained model and parameters"""
		self.params = json.loads(open(parameters).read())

		theme_dic = cPickle.load(open(theme_dic, 'rb'))
		self.inv_theme_dic = {}
		for key,num in theme_dic.items():
			self.inv_theme_dic[num] = key

		self.coord = tf.train.Coordinator()

		self.sess = tf.Session()


		self.saver = tf.train.import_meta_graph(model+'.meta')
		self.saver.restore(self.sess,tf.train.latest_checkpoint('./models/'))

		self.graph = tf.get_default_graph()

		self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
		
		self.doc_batch = self.graph.get_tensor_by_name("document_batch:0")
		self.sentlen_batch = self.graph.get_tensor_by_name("sentlen_batch:0")
		self.doclen_batch = self.graph.get_tensor_by_name("doclen_batch:0")
		self.keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
		self.document_vector = self.graph.get_tensor_by_name("AttentionBasedAggregator/document-vectorss:0")
#		self.labels_batch = self.graph.get_tensor_by_name("labels_batch:0")

		self.predict_ldcsf = self.graph.get_tensor_by_name("Ldcsf-Att-Prediction-Layer/predict_ldcsf_1:0")


		self.ldcloader = LdcTestLoader()

	def test(self,filename):

		doc_batch,doclen_batch,sentlen_batch = self.ldcloader.xml2batch(doc_file=filename)

		with self.sess.as_default():
			predict_ldcsf = self.sess.run([self.predict_ldcsf], feed_dict={self.doc_batch: doc_batch,
				self.sentlen_batch: sentlen_batch, self.doclen_batch: doclen_batch,
				self.keep_prob : 1.0 })

			predict_ldcsf = predict_ldcsf[0].tolist()[0]
			predict_ldcsf = list(map(int, predict_ldcsf))
			print(predict_ldcsf)

			predict_indexes = [i for i, e in enumerate(predict_ldcsf) if e != 0]

			print("\nPREDICTED THEMES :")
			for index in predict_indexes:
				print(self.inv_theme_dic[index])
#		summary = tf.Summary(value=[tf.Summary.Value(tag="LDCSF-DocClassifier-validloss",
#			simple_value=float(loss_ldcsf))])




		
