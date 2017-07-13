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
from lib.util import *
from lib.path import *


def generate_batch_data_mono_skip(skip_window=5):

	data_mono = cPickle.load(open(DATA_ID+"mono.p", 'rb'))
	
	batch_mono = open(DATA_BATCH+"mono.csv",'w')

	for sent in data_mono:
		for j in range(skip_window):
			sent = ['<eos>'] + sent + ['<eos>']
		for j in range(skip_window,len(sent)-skip_window):
			for skip in range(1,skip_window+1):
				if sent[j-skip] != '<eos>':
					batch_mono.write(str(sent[j])+","+str(sent[j-skip])+"\n")
				if sent[j+skip] != '<eos>':
					batch_mono.write(str(sent[j])+","+str(sent[j+skip])+"\n")
	batch_mono.close()

def generate_batch_data_multi_skip(window = 5):

	data_bi = cPickle.load(open(DATA_ID+"bi_train.p", 'rb'))
	
	batch_bi = open(DATA_BATCH+"bi_train.csv",'w')

	for sent_pair in data_bi:
		sent1 = sent_pair[0]
		sent2 = sent_pair[0]

		sent1_len = float(len(sent1))
		sent2_len = float(len(sent2))

		for j in range(len(sent1)):

			alignment = int((j/sent1_len) * sent2_len)
			window_high = alignment + window
			window_low = alignment - window
			if window_low < 0:
				window_low = 0

			for k in sent2[alignment:window_high]:

				# l1 -> l2
				batch_bi.write(str(sent1[j])+","+str(k)+"\n")

				# l2 -> l1
				batch_bi.write(str(k)+","+str(sent1[j])+"\n")

			for k in sent2[window_low:alignment]:
				# l1 -> l2
				batch_bi.write(str(sent1[j])+","+str(k)+"\n")

				# l2 -> l1
				batch_bi.write(str(k)+","+str(sent1[j])+"\n")


def generate_batch_data_task_sentsim(max_length=50, neg_sample = 1):
	''' 
	generate batch for sentence similarity

	this stores data as

	sent1 sent2 sent1len sent2len 1 (1 because they are parallel/similar sentences)
	sent1 randsent sent1len randsent_len 0 (0 because they are not similar sentences)
	sent2 randsent sent2len randsent_len 0
	'''

	data_bi = cPickle.load(open(DATA_ID+"bi_train.p", 'rb'))
	data_mono = cPickle.load(open(DATA_ID+"mono.p", 'rb'))

	# batch file for training sentence similarity
	batch_sentsim = open(DATA_BATCH+"sentsim.csv",'w')
	for pair in data_bi:

		# make length of each sentence to max length
		sent1_len = str(len(pair[0]))
		sent2_len = str(len(pair[1]))
		pair[0] = pad(pair[0][:max_length], 0, max_length)
		pair[1] = pad(pair[1][:max_length], 0, max_length)
		sent1 = ",".join(str(x) for x in pair[0])
		sent2 = ",".join(str(x) for x in pair[1])

		batch_sentsim.write(sent1+","+sent2+","+sent1_len+","+sent2_len+",1\n")

		for i in range(0,neg_sample):

			# we add a random sentence from monolingual sentence to say, that it's not similar to it
			rand_mono = random.choice(data_mono)
			rand_mono_len = str(len(rand_mono))
			rand_mono = pad(rand_mono[:max_length], 0, max_length)
			negative = ",".join(str(x) for x in rand_mono)

			batch_sentsim.write(sent1+","+negative+","+sent1_len+","+rand_mono_len+",0\n")
			batch_sentsim.write(sent2+","+negative+","+sent2_len+","+rand_mono_len+",0\n")

	batch_sentsim.close()

	del data_bi #saving memory

	data_valid = cPickle.load(open(DATA_ID+"bi_valid.p", 'rb'))

	print(data_valid.keys())
	for key in data_valid.keys():
		#filename of the valid file
		filename = "valid_"+key.replace(":","_")+".csv"

		batch_valid = open(DATA_BATCH+filename,'w')

		for pair in data_valid[key]:

			sent1_len = str(len(pair[0]))
			sent2_len = str(len(pair[1]))
			pair[0] = pad(pair[0][:max_length], 0, max_length)
			pair[1] = pad(pair[1][:max_length], 0, max_length)
			sent1 = ",".join(str(x) for x in pair[0])
			sent2 = ",".join(str(x) for x in pair[1])

			batch_valid.write(sent1+","+sent2+","+sent1_len+","+sent2_len+",1\n")

			# we add a random sentence from monolingual sentence to say, that it's not similar to it
			rand_mono = random.choice(data_mono)
			rand_mono_len = str(len(rand_mono))
			rand_mono = pad(rand_mono[:max_length], 0, max_length)


			negative = ",".join(str(x) for x in rand_mono)

			batch_valid.write(sent1+","+negative+","+sent1_len+","+rand_mono_len+",0\n")
			batch_valid.write(sent2+","+negative+","+sent2_len+","+rand_mono_len+",0\n")

		batch_valid.close()

##################### TED CORPUS BATCH DATA GENERATOR #####################

#### ---- helper function for creating batch data  for ted corpus ---- ####

def random_document_picker(ted_corpus,mode='train'):
	random_lang1 = random.choice(list(ted_corpus.keys()))
	random_lang2 = random.choice(list(ted_corpus[random_lang1].keys()))

	corpus = ted_corpus[random_lang1][random_lang2][mode]

	random_key = random.choice(list(corpus.keys()))
	random_key2 = random.choice(list(corpus[random_key].keys()))

	random_filename = random.choice(list(corpus[random_key][random_key2].keys()))

	return corpus[random_key][random_key2][random_filename]

def random_sentence_picker(ted_corpus,mode='train'):
	random_lang1 = random.choice(list(ted_corpus.keys()))
	random_lang2 = random.choice(list(ted_corpus[random_lang1].keys()))

	corpus = ted_corpus[random_lang1][random_lang2][mode]

	random_key = random.choice(list(corpus.keys()))
	random_key2 = random.choice(list(corpus[random_key].keys()))

	random_filename = random.choice(list(corpus[random_key][random_key2].keys()))

	while len(corpus[random_key][random_key2][random_filename]) == 0:
		random_filename = random.choice(list(corpus[random_key][random_key2].keys()))

	random_sent = random.choice(corpus[random_key][random_key2][random_filename])
	
	return random_sent

#--------------------------------------------------------------#

def generate_batch_data_task_docsim(langpair=['en-de'], max_sent_len = 50, max_doc_size = 50):

	'''
	generates the data in 
	doc1 doc2 doc3 doc4 format, where
	doc1 doc2 are similar
	doc3 is a document with same number of lines as doc1, doc2 (for sentence level loss)
	doc4 is a random document which is not similar to doc1/doc2

	NOTE : make sure you have ted.p exisiting at DATA_ID folder
		   check path.py to set right path for DATA_ID
	'''

	ted_corpus = cPickle.load(open(DATA_ID+"ted.p", 'rb'))

	print("ted corpus loaded")

	lang1 = langpair[0].split('-')[0]
	lang2 = langpair[0].split('-')[1]

	# get lang1 train corpus
	lang1_train_corpus = ted_corpus[lang1][lang2]['train']
	lang1_train_corpus_keys = list(lang1_train_corpus.keys())
	random.shuffle(lang1_train_corpus_keys)

	print(lang1_train_corpus_keys)
	print("Creating Epoch data for TED document similarity")
	# get lang2 train corpus
	lang2_train_corpus = ted_corpus[lang2][lang1]['train']

	# randomly pick random category
	epoch = []
	sample_count = 0
	for key in lang1_train_corpus_keys:

		lang1_train_corpus_key_keys = list(lang1_train_corpus[key].keys())
		random.shuffle(lang1_train_corpus_key_keys)

		# randomly pick positive / negative
		for key2 in lang1_train_corpus_key_keys:

			for filename in lang1_train_corpus[key][key2]:

				if filename in lang2_train_corpus[key][key2].keys():

					sample_count = sample_count + 1
					document_len = []
					sequence_length = []

					#document 1 
					doc1 = lang1_train_corpus[key][key2][filename]
					if len(doc1) > max_doc_size:
						doc1len = max_doc_size
						doc3len = doc1len
					else:
						doc1len = len(doc1)
						doc3len = doc1len
						assert doc1len != 0
					document_len.append(doc1len)

					sent_length = []
					for line in doc1:
						sent_length.append(len(line))
					doc1, sent_length = document_pad(doc1, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					sequence_length.append(sent_length)

					#document 2
					doc2 = lang2_train_corpus[key][key2][filename]
					if len(doc2) > max_doc_size:
						doc2len = max_doc_size
					else:
						doc2len = len(doc2)
					document_len.append(doc2len)

					sent_length = []
					for line in doc2:
						sent_length.append(len(line))
					doc2, sent_length = document_pad(doc2, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					sequence_length.append(sent_length)

					#document 3 : negative sentences
					doc3 = []
					sent_length = []
					for i in range(0,doc3len):
						random_sent = random_sentence_picker(ted_corpus)
						doc3.append(random_sent)
						sent_length.append(len(random_sent))

					if doc3len > max_doc_size:
						doc3len = max_doc_size
					document_len.append(doc3len)

					doc3, sent_length = document_pad(doc3, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					sequence_length.append(sent_length)

					#document 4 : negative document
					doc4 = random_document_picker(ted_corpus,mode='train')
					if len(doc4) > max_doc_size:
						doc4len = max_doc_size
					else:
						doc4len = len(doc4)
					document_len.append(doc4len)

					sent_length = []
					for line in doc4:
						sent_length.append(len(line))
					doc4, sent_length = document_pad(doc4, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
					assert len(doc4) != 0
					sequence_length.append(sent_length)


					sample = [doc1] + [doc2] + [doc3] + [doc4] + [document_len] + [sequence_length]
					epoch.append(sample)
	print("Data Created : total samples",sample_count)
	return epoch


#############################################################################
############## LEIDOS CLASSIFICATION DATA BATCH GENERATOR ###################

def generate_train_batch_data_task_leidos(max_sent_len=50,max_doc_size=100):

	'''
	make sure this leidos_train.p exists in DATA_ID folder
	1. This function takes leidos_train.p which is a dictionary
	with keys as document ID.
	2. For each key, we concat 'tokens_title' and 'tokens_text'
	to make a list of sentences ( list of list)
	3. Labels is already a list where theme(s) are represented as 1
	4. We also pad all sentences to max_sent_len
	5. Data is stored into leidos_train.csv in the following format
	
	doc,sent_length,labels
	doc : list of sentences [100*50]
	sent_length = [100]
	labels = [<number of labels>]
	'''
	leidos_corpus = cPickle.load(open(DATA_ID+"leidos_train.p", 'rb'))

	epoch = []

	for key in leidos_corpus.keys():
		sample = []
		labels = leidos_corpus[key]['theme']		
		doc = leidos_corpus[key]['tokens_title'] + leidos_corpus[key]['tokens_text']
		doc_len = len(doc)
		'''
		1. if the doc_len is more than max_doc_size, then we only take
		sentences max_doc_size
		2. also check if there is an empty document
		'''
		if doc_len > max_doc_size:
			doc_len = max_doc_size
		else:
			assert doc_len != 0

		sent_length = []
		for line in doc:
			sent_length.append(len(line))
		doc,sent_length = document_pad(doc, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
		sample.append(doc)
		sample.append(doc_len)
		sample.append(sent_length)
		sample.append(labels)
		epoch.append(sample)

	return epoch

def generate_test_batch_data_task_leidos(max_sent_len=50,max_doc_size=100):

	leidos_corpus = cPickle.load(open(DATA_ID+"leidos_test.p", 'rb'))

	epoch = {}
	for lang in leidos_corpus.keys():
		
		epoch[lang] = []
		for key in leidos_corpus[lang].keys():
			sample = []
			labels = leidos_corpus[lang][key]['theme']
			doc = leidos_corpus[lang][key]['tokens_title'] + leidos_corpus[lang][key]['tokens_text']
			doc_len = len(doc)
			'''
			1. if the doc_len is more than max_doc_size, then we only take
			sentences max_doc_size
			2. also check if there is an empty document
			'''
			if doc_len > max_doc_size:
				doc_len = max_doc_size
			else:
				assert doc_len != 0

			sent_length = []
			for line in doc:
				sent_length.append(len(line))
			doc,sent_length = document_pad(doc, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
			
			sample.append(doc)
			sample.append(doc_len)
			sample.append(sent_length)
			sample.append(labels)
			epoch[lang].append(sample)

	return epoch

def _onehot2binarylabels(labels):

	binary_labels = []
	for i in range(0,len(labels)):
		if labels[i] == 0:
			binary = [0,1]
		else:
			binary = [1,0]

		binary_labels.append(binary)

	return binary_labels
#############################################################################

#############################################################################
############## LDC SF CLASSIFICATION DATA BATCH GENERATOR ###################

def generate_batch_data_task_ldcsf(filename=DATA_ID+"sec_pilot_train.p",
	max_sent_len=50,max_doc_size=100):

	'''
	make sure this leidos_train.p exists in DATA_ID folder
	1. This function takes leidos_train.p which is a dictionary
	with keys as document ID.
	2. For each key, we concat 'tokens_title' and 'tokens_text'
	to make a list of sentences ( list of list)
	3. Labels is already a list where theme(s) are represented as 1
	4. We also pad all sentences to max_sent_len
	5. Data is stored into leidos_train.csv in the following format
	
	doc,sent_length,labels
	doc : list of sentences [100*50]
	sent_length = [100]
	labels = [<number of labels>]
	'''
	leidos_corpus = cPickle.load(open(filename, 'rb'))

	epoch = []

	for key in leidos_corpus.keys():
		sample = []
		labels = leidos_corpus[key]['theme']		
		doc = leidos_corpus[key]['tokens_title'] + leidos_corpus[key]['tokens_text']
		doc_len = len(doc)
		'''
		1. if the doc_len is more than max_doc_size, then we only take
		sentences max_doc_size
		2. also check if there is an empty document
		'''
		if doc_len > max_doc_size:
			doc_len = max_doc_size
		else:
			assert doc_len != 0

		sent_length = []
		for line in doc:
			sent_length.append(len(line))
		doc,sent_length = document_pad(doc, 0, max_sent_len=max_sent_len, doc_length=max_doc_size, sent_length=sent_length)
		sample.append(doc)
		sample.append(doc_len)
		sample.append(sent_length)
		sample.append(labels)
		epoch.append(sample)

	return epoch





		
