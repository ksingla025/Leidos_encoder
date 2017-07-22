#!/usr/bin/python

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
import codecs
import json
import nltk

#from utils.twokenize import *
from lib.path import *

def preprocess_text(text):
	
	text = text.strip()
	text = nltk.word_tokenize(text)
#	text = tokenizeRawTweetText(text)
	text = ' '.join(text)
	text = text.lower()
#	text = text.lower()

	return text

# data realtied class and batch/batch-data generator functions
class DataBuilder(object):
	''' 
	1. this class reads data from monolingual + parallel files
	2. cleans them : read_data(lang_ext=1)
	3. makes dictionary, replace words with integer IDs : 
	build_dataset(bilangs, min_count)
	'''
	def __init__(self, lang_ext=1, min_count=5, data_usage=['mono','bi','ted']):

		self.lang_ext = lang_ext
		self.min_count = min_count # minimum count of each word in each language
		self.data_usage = data_usage

	def read_data(self):
		"""Extract the first file enclosed in a zip file as a list of words"""

		# cleaning monolingual files and dump clean files
		all_langs = []
		mono_files = glob.glob(DATA_MONO + '*')

		
		for filename in mono_files:
			print(filename)
			lang = filename.split('.')[-1]
			if lang not in all_langs:
				all_langs.append(lang)
			ext = ':ID:' + lang
			out_file = open(DATA_MONO_CLEAN + os.path.basename(filename) + '.cl','w')
			with open(filename) as infile:
				for line in infile:
					line = preprocess_text(line)
					if line != '':			
						# lang_ext is sticked to each token
						if self.lang_ext == 1:
							tokens = [x + ext for x in line.split()]
						else:
							tokens = line.split()
						tokens = " ".join(tokens)
						out_file.write(tokens + "\n")
			out_file.close()


		# cleanining bilingual files and dump clean files
		self.bilangs = []
		bi_files = glob.glob(DATA_BI + '*')
		for filename in bi_files:
			print(filename)
			count = 0

			src = filename.split('.')[-1].split('-')[0]
			tgt = filename.split('.')[-1].split('-')[1]
			src_lang = ':ID:' + src
			tgt_lang = ':ID:' + tgt

			if DATA_BI_CLEAN + os.path.basename(filename) not in self.bilangs:
				self.bilangs.append(DATA_BI_CLEAN + os.path.basename(filename))

			out_src_file = open(DATA_BI_CLEAN + os.path.basename(filename) + 
				'.'+ src + '.cl','w')
			out_tgt_file = open(DATA_BI_CLEAN + os.path.basename(filename) + 
				'.'+ tgt + '.cl','w')

			with open(filename) as sentence_pair_file:
			
				for sentence_pair_line in sentence_pair_file:
					sentence_pair_line = sentence_pair_line.rstrip()
				
					if len(sentence_pair_line.split(' ||| ')) == 2:
						source_line, target_line = sentence_pair_line.split(' ||| ')

						source_line = preprocess_text(source_line)
						target_line = preprocess_text(target_line)
						count = count + 1
						if source_line != '' and target_line != '':
							source_tokens, target_tokens = source_line.split(' '), target_line.split(' ')
							if self.lang_ext == 1:
								source_tokens = [x + src_lang for x in source_tokens]
								target_tokens = [x + tgt_lang for x in target_tokens]
							source_tokens = ' '.join(source_tokens)
							target_tokens = ' '.join(target_tokens)
							out_src_file.write(source_tokens + "\n")
							out_tgt_file.write(target_tokens + "\n")
			print(count)
			out_src_file.close()
			out_tgt_file.close()

	def create_dictionaries(self):
		''' creates dictionary using monolingual, bilingual, ted corpus
		'''

		wordcount = {}
		wordcount_ted = {}


		# create counter from monolingual data
		if 'mono' in self.data_usage:
			mono_files = glob.glob(DATA_MONO_CLEAN + '*')
			for filename in mono_files:
				print(filename)
				lang = os.path.basename(filename).split('.')[-2]
				file = open(filename,'r')
				wordcount[lang] = Counter(file.read().split())
#				wordcount[lang] = {k:v for k, v in wordcount[lang].items() if v > self.min_count}
				file.close()
			print('counter created from mono-files')


		# update counter from bilingual
		if 'bi' in self.data_usage:
			bi_files = glob.glob(DATA_BI_CLEAN + '*')
			for filename in bi_files:
				print(filename)
				lang = os.path.basename(filename).split('.')[-2]
				file = open(filename,'r')
#				file_counter = {k:v for k, v in Counter(file.read().split()).items() if v > self.min_count}
				wordcount[lang] = wordcount[lang] + Counter(file.read().split())
				file.close()
			print('counter created from bi-files')

		# update counter from TED corpus
		# find all folders in ted folder, they are named according to language pair, en-de,de-en
		if 'ted' in self.data_usage:
			ted_folders = glob.glob(os.path.join(DATA_TED_CLEAN, '*'))
#			ted_folders = os.listdir(DATA_TED_CLEAN)
			print(ted_folders)

			for ted_folder in ted_folders:

				lang = os.path.basename(ted_folder).split('-')[0]

				if lang not in wordcount_ted.keys():
					wordcount_ted[lang] = Counter()

				train_path = ted_folder + '/train'
#				train = os.listdir(train_path)
				train = glob.glob(os.path.join(train_path, '*'))
				print(train)

				for category in train:
					dirss = glob.glob(os.path.join(category, '*'))
					print(dirss)
					for dirs in dirss:
						files = glob.glob(dirs + '/*')
						for file in files:
							file = open(file,'r')
							wordcount_ted[lang] = wordcount_ted[lang] + Counter(file.read().split())
							file.close()
			print('counter created from TED files')
	
		#merge wordcount of ted and mono+bi corpus
		for lang in wordcount_ted.keys():
			print('Merging wordcount for language :',lang)
			wordcount[lang] = wordcount[lang] + wordcount_ted[lang]	

		dictionary = dict() # {word : index}
		for lang in wordcount.keys():
			#remove key:value with freq < self.min_count
			print(len(wordcount[lang]),lang)
			wordcount[lang] = {k:v for k, v in wordcount[lang].items() if v > self.min_count}
			# adding words to dictionaries
			for word in wordcount[lang]:
				dictionary[word] = len(dictionary)
		del wordcount
		print('dictionary created')
		print('Dictionary size',len(dictionary.keys()))

		cPickle.dump(dictionary, open(DATA_ID + 'dictionary.p', 'wb'))

		reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
		cPickle.dump(reverse_dictionary, open(DATA_ID + 'reverse_dictionary.p', 'wb'))

	def build_dataset(self):
		'''
		Build the dictionary and replace rare words with UNK token.
		1. we make sure there is equal representation of data in dev sets
		Parameters
		----------
		words: list of tokens
		vocabulary_size: maximum number of top occurring tokens to produce, 
			rare tokens will be replaced by 'UNK'/0
		'''
		print('Build Dataset and dictionaries')

		# counter for making dictionary	
		
		dictionary = cPickle.load(open(DATA_ID + 'dictionary.p', 'rb'))
		
		mono_files = glob.glob(DATA_MONO_CLEAN + '*')
		bi_files = glob.glob(DATA_BI_CLEAN + '*')



		## replace words by IDs in monolingual data
		if 'mono' in self.data_usage:
			data_mono = list()
			for filename in mono_files:
				file = open(filename,'r')
				for line in file:
					line = line.strip().split()
					for i in range(0,len(line)):
						if line[i] in dictionary:
							index = dictionary[line[i]]
						else:
							index = 0
						line[i] = index
					data_mono.append(line)

			random.shuffle(data_mono)
			cPickle.dump(data_mono, open(DATA_ID + 'mono.p', 'wb'))
			del data_mono
			print('mono data created')


		## replace words by IDs in bilingual data
		if 'bi' in self.data_usage:
			data_bi_train = list()
			data_bi_valid = {}
			data_bi_test = {}
			for filename in self.bilangs:
				bi_temp = list()
				print(filename)
				lang1 = os.path.basename(filename).split('.')[1].split('-')[0]
				lang2 = os.path.basename(filename).split('.')[1].split('-')[1]

				lang1_file = open(filename+ '.' + lang1 + '.cl').readlines()
				lang2_file = open(filename+ '.' + lang2 + '.cl').readlines()

				sent_pair = []
				for i in range(0,len(lang1_file)):
					sent_pair = [lang1_file[i].split(), lang2_file[i].split()]
					pair = []
					for seq in sent_pair:
						for i in range(0,len(seq)):
							if seq[i] in dictionary:
								index = dictionary[seq[i]]
							else:
								index = 0 # dictionary['UNK']
							seq[i] = index
						pair.append(seq)
					bi_temp.append(pair)

				random.shuffle(bi_temp)

				# train data is kept together
				data_bi_train = data_bi_train + bi_temp[:int(.9*len(bi_temp))]

				#validation and test data is stored according to language pairs
				data_bi_valid[lang1+":"+lang2] = bi_temp[int(.9*len(bi_temp)):int(.95*len(bi_temp))]
				data_bi_test[lang1+":"+lang2] = bi_temp[int(.95*len(bi_temp)):]

				del lang1_file
				del lang2_file

			random.shuffle(data_bi_train)
			cPickle.dump(data_bi_train, open(DATA_ID + 'bi_train.p', 'wb'))
			cPickle.dump(data_bi_valid, open(DATA_ID + 'bi_valid.p', 'wb'))
			cPickle.dump(data_bi_test, open(DATA_ID + 'bi_test.p', 'wb'))

			del data_bi_train # saving memory
			del data_bi_valid # saving memory
			del data_bi_test # saving memory

			print('bi data created')

		## replace words in ted-files by ID's
		if 'ted' in self.data_usage:

			ted = {}

			ted_folders = glob.glob(os.path.join(DATA_TED_CLEAN, '*'))

			print(ted_folders)

			for ted_folder in ted_folders:

				lang1 = os.path.basename(ted_folder).split("-")[0]
				lang2 = os.path.basename(ted_folder).split("-")[1]

				if lang1 not in ted.keys():
					ted[lang1] = {}

				if lang2 not in ted[lang1].keys():
					ted[lang1][lang2] = {}

				ted[lang1][lang2]['train'] = {}
				ted[lang1][lang2]['test'] = {}

				for key in ted[lang1][lang2].keys():
					# create train-files
					train_path = ted_folder + '/' + key
					train = glob.glob(os.path.join(train_path, '*'))
					print(train)

					# read each category in train arts/education etc
					for category in train:
						category_name = os.path.basename(category)
						ted[lang1][lang2][key][category_name] = {}
						dirss = glob.glob(os.path.join(category, '*'))
						print(dirss)

						# negative and positive dirs in like. arts
						for dirs in dirss:
							dir_name = os.path.basename(dirs)
							ted[lang1][lang2][key][category_name][dir_name] = {}

							# read each file in dirs ( negative / positive)
							files = glob.glob(dirs + '/*')
							for file in files:
#								print file
								file_name = os.path.basename(file)
								ted[lang1][lang2][key][category_name][dir_name][file_name] = []

								# replace words in each file by the dictionary IDs
								file = open(file,'r').readlines()
								for i in range(0,len(file)):
									line = file[i].split()
									for j in range(0,len(line)):
										if line[j] in dictionary:
											line[j] = dictionary[line[j]]
										else:
											line[j] = 0
									ted[lang1][lang2][key][category_name][dir_name][file_name].append(line)

								#check if the file is empty or not
								print(len(ted[lang1][lang2][key][category_name][dir_name][file_name]))
							
								if len(ted[lang1][lang2][key][category_name][dir_name][file_name]) == 0:
									print('deleting',ted[lang1][lang2][key][category_name][dir_name][file_name])
									ted[lang1][lang2][key][category_name][dir_name].pop(file_name, None)


			print('TED files created')

			cPickle.dump(ted, open(DATA_ID + 'ted.p', 'wb'))


##### utility function for both leidos and ldc_sf corpus ######
def tokens_text_to_index(tokens_text, lang_ext, dictionary):
	''' 
	tokens text is a list of lines, we just replace each word by the dictionary index
	'''
	processed_text = []

	for line in tokens_text:
		line = " ".join(line)
		line = preprocess_text(line)
		line = line.split()
		for i in range(0,len(line)):
			line[i] = line[i] + ":ID:" + lang_ext
			if line[i] in dictionary:
				line[i] = dictionary[line[i]]
			else:
				line[i] = 0
		processed_text.append(line)

	return processed_text

class YelpCorpus(object):

	def __init__(self,theme_num=5, dictionary=DATA_ID+"dictionary.p"):

		self.dictionary = cPickle.load(open(dictionary, 'rb'))
		#theme_dic is used to keep counter of themes
		self.theme_num = theme_num
		self.theme_dic = {}
		self.theme_counter = 0

	def build_dataset(self, lang_ext = 'en'):
		'''
		there should be a file tokenized_en_train.json in DATA_LEIDOS folder
		'''
		print("Loading Yelp dataset")
		data = self._load_json_file(filename=DATA_YELP+"yelp_toy.json", lang_ext="en")
		print("theme counter", self.theme_counter)

		print("data_fetched")
		random.shuffle(data)
		print("shuffled data")

		train_bound = int(len(data)*0.8)
		train = data[:train_bound]
		test = data[train_bound:len(data)]
		cPickle.dump(train, open(DATA_ID + 'yelp_train.p', 'wb'))
		cPickle.dump(train, open(DATA_ID + 'yelp_test.p', 'wb'))
		cPickle.dump(self.theme_dic, open(DATA_ID + 'yelp_theme_dic.p', 'wb'))

	def _load_json_file(self,filename,lang_ext="en"):

		file_data = []
		f = codecs.open(filename,"r","utf-8")
		done = False
		while not done:

			line = f.readline()
			if line == '':
				done = True
			else:
				file_line = []
				data = json.loads(line)

				data["text"] = nltk.sent_tokenize(data["text"])
				for i in range(len(data["text"])):
					data["text"][i] = data["text"][i].split()
				index_tokens_text = tokens_text_to_index(data["text"], lang_ext=lang_ext,
					dictionary=self.dictionary)
				file_line.append(index_tokens_text)
				file_line.append(self._themelist_to_onehot(data["stars"]))
				#if theme exists in data line
				file_data.append(file_line)

		return file_data
	
	def _themelist_to_onehot(self,theme):
		'''
		it takes a list of themes for a sample document and then convert it to one hot
		'''
		theme_vector = [0]*self.theme_num
		if theme not in self.theme_dic.keys():
			self.theme_dic[theme] = self.theme_counter
			self.theme_counter = self.theme_counter + 1
		theme_vector[self.theme_dic[theme]] = 1

		return theme_vector


# iterable corpus
class LeidosCorpus(object):

	def __init__(self, theme_num=50, dictionary = DATA_ID+"dictionary.p"):

		self.dictionary = cPickle.load(open(dictionary, 'rb'))
		#theme_dic is used to keep counter of themes
		self.theme_num = theme_num
		self.theme_dic = {}
		self.theme_counter = 0

	def build_dataset_train(self, lang_ext = 'en'):
		'''
		there should be a file tokenized_en_train.json in DATA_LEIDOS folder
		'''
		data = self._load_json_file(filename=DATA_LEIDOS+"tokenized_"+lang_ext+"_train.json", lang_ext=lang_ext)
		cPickle.dump(data, open(DATA_ID + 'leidos_train.p', 'wb'))
		cPickle.dump(self.theme_dic, open(DATA_ID + 'leidos_theme_dic.p', 'wb'))

	def build_dataset_test(self, lang_ext = ['en','es','fr']):
		'''
		there should be files :
		tokenized_en_test.json
		tokenized_es_test.json #labels/themes are missing, get labels from tokenized_es_test_english.json
		tokenized_fr_test.json #labels/themes are missing, get labels from tokenized_fr_test_english.json 
		'''
		test_data = {}
		for lang in lang_ext:

			if lang == 'en':			
				data = self._load_json_file(filename=DATA_LEIDOS+"tokenized_en_test.json",lang_ext=lang)
				test_data[lang] = data
			else:
				# en & fr data has no labels
				data = self._load_json_file(filename=DATA_LEIDOS+"tokenized_"+lang+"_test.json",lang_ext=lang)
				data_with_theme = self._load_json_file(filename=DATA_LEIDOS+"tokenized_"+lang+"_test_english.json",lang_ext=lang)
				data = self._add_theme_to_file(nothemefile=data, themefile=data_with_theme)
				test_data[lang] = data

		cPickle.dump(test_data, open(DATA_ID + 'leidos_test.p', 'wb'))

	def _load_json_file(self,filename,lang_ext):

		file_data = {}
		f = codecs.open(filename,"r","utf-8")
		done = False
		while not done:

			line = f.readline()
			if line == '':
				done = True
			else:
				data = json.loads(line)
				file_data[data["id"]] = {}

				index_tokens_text = tokens_text_to_index(data["tokens_text"], lang_ext=lang_ext,
					dictionary=self.dictionary)
				file_data[data["id"]]["tokens_text"] = index_tokens_text

				index_tokens_title = tokens_text_to_index(data["tokens_title"], lang_ext=lang_ext,
					dictionary=self.dictionary)
				file_data[data["id"]]["tokens_title"] = index_tokens_title

				#if theme exists in data line
				if "theme" in data.keys():
					if len(data['theme']) == 0:
						print("No theme")
					file_data[data["id"]]["theme"] = self._themelist_to_onehot(data["theme"])



		return file_data

	def _add_theme_to_file(self, nothemefile, themefile):
		''' 
		takes two files (output of load_json_file function), one file has a key them for each fileID and one doesn't
		it just takes themes from themefile and appends it the say fileID dictionaries
		1. also removes files which are not there in themefile/test-data
		'''
		for key in themefile.keys():
			if key in nothemefile.keys():
				themefile[key]['tokens_text'] = nothemefile[key]['tokens_text']
				themefile[key]['tokens_title'] = nothemefile[key]['tokens_title']
			else:
				del themefile[key]

		return themefile

	def _themelist_to_onehot(self,themelist):
		'''
		it takes a list of themes for a sample document and then convert it to one hot
		'''
		theme_vector = [0]*self.theme_num
		for theme in themelist:
			if theme not in self.theme_dic.keys():
				self.theme_dic[theme] = self.theme_counter
				self.theme_counter = self.theme_counter + 1
			theme_vector[self.theme_dic[theme]] = 1

		return theme_vector

class LDCSF_Corpus(object):
	'''
	builder class for converting LDC SF corpus into word_indexes and also theme into 50 dimentional vectors
	'''
	def __init__(self, theme_num=12, dictionary = DATA_ID+"dictionary.p"):

		self.dictionary = cPickle.load(open(dictionary, 'rb'))

		#theme_dic is used to keep counter of themes
		self.theme_num = theme_num
		self.theme_dic = {}
		self.theme_counter = 0

	def build_dataset_train(self, lang_ext = 'en'):
		'''
		there should be a file sec_pilot_english_train.json in DATA_LDC_SF folder
		'''
		data = self._load_json_file(filename=DATA_LDC_SF+"sec_pilot.json", lang_ext=lang_ext)
		cPickle.dump(data, open(DATA_ID + 'sec_pilot_train.p', 'wb'))
		cPickle.dump(self.theme_dic, open(DATA_ID + 'ldcsf_theme_dic.p', 'wb'))

	def build_dataset_test(self, lang_ext = 'en'):
		'''
		there should be a file sec_pilot_english_train.json in DATA_LDC_SF folder
		'''
		data = self._load_json_file(filename=DATA_LDC_SF+"sec_pilot_evaluation.json", lang_ext=lang_ext)
		cPickle.dump(data, open(DATA_ID + 'sec_pilot_eval.p', 'wb'))



	def _load_json_file(self,filename,lang_ext):

		file_data = {}
		f = codecs.open(filename,"r","utf-8")
		done = False
		while not done:

			line = f.readline()
			if line == '':
				done = True
			else:
				data = json.loads(line)
				print(data)
				index_tokens_text = tokens_text_to_index(data["tokens_text"], lang_ext=lang_ext,
					dictionary=self.dictionary)
				print(index_tokens_text)

				index_tokens_title = tokens_text_to_index(data["tokens_title"], lang_ext=lang_ext,
					dictionary=self.dictionary)
				#if theme exists in data line
				if len(data['theme']) == 0:
						print("No theme")
				else:
					file_data[data["id"]] = {}
					if "need_type" in data["theme"]:
						data["theme"].remove("need_type")
					file_data[data["id"]]["theme"] = self._themelist_to_onehot(themelist=data["theme"])
					print(file_data[data["id"]]["theme"])
					file_data[data["id"]]["tokens_title"] = index_tokens_title
					file_data[data["id"]]["tokens_text"] = index_tokens_text

		return file_data


	def _themelist_to_onehot(self,themelist):
		'''
		it takes a list of themes for a sample document and then convert it to one hot
		'''
		theme_vector = [0]*self.theme_num
		for theme in themelist:
			if theme not in self.theme_dic.keys():
				self.theme_dic[theme] = self.theme_counter
				self.theme_counter = self.theme_counter + 1
			theme_vector[self.theme_dic[theme]] = 1

		return theme_vector







