#!/usr/bin/python

import xmltodict
from collections import OrderedDict
import json
import codecs
import nltk
import _pickle as cPickle

import numpy as np

from lib.databuilder import tokens_text_to_index
from lib.util import *
from lib.path import *
import re

debug = False

need_frame_types = [
	"Evacuation",
	"Food Supply",
	"Search/Rescue",
	"Utilities, Energy, or Sanitation",
	"Infrastructure",
	"Medical Assistance",
	"Shelter",
	"Water Supply"
]
issue_frame_types = [
	"Civil Unrest or Wide-spread Crime",
	"Regime Change",
	"Terrorism or other Extreme Violence"
]


frame_translation = {
	"evac" : "Evacuation",
	"food" : "Food Supply",
	"search" : "Search/Rescue",
	"utils" : "Utilities, Energy, or Sanitation",
	"infra" : "Infrastructure",
	"med" : "Medical Assistance",
	"shelter" : "Shelter",
	"water" : "Water Supply",
	"a" : "Civil Unrest or Wide-spread Crime",
	"a" : "Regime Change",
	"crimeviolence" : "Terrorism or other Extreme Violence"
}

def leidos_xml2dic(doc_file):

	doc_output = {}
	doc_id = re.sub("\.ltf\.xml$","",doc_file)
	doc_output['id'] = doc_id
	doc_output["tokens_text"] = []

	# collect the text
	lines = []
	fd = codecs.open(doc_file, "r", "utf-8")

	xml_corpus = xmltodict.parse(fd.read())

	fd.close()

	xml_documents = xml_corpus['LCTL_TEXT']['DOC']['TEXT']['SEG']

	if type(xml_documents) == OrderedDict:
		xml_documents = [xml_documents]
	for seg in xml_documents:
		lines.append( nltk.word_tokenize(seg['ORIGINAL_TEXT']) )
	doc_output["tokens_text"] = lines

	return doc_output

class LdcTestLoader(object):

	def __init__(self, theme_num=12, dictionary = DATA_ID+"dictionary.p",
		theme_dic = DATA_ID + 'ldcsf_theme_dic.p'):

		self.theme_num = theme_num
		self.dictionary = cPickle.load(open(dictionary, 'rb'))
		self.theme_dic = cPickle.load(open(theme_dic, 'rb'))

	def xml2batch(self, doc_file,max_sent_len=50,max_doc_size=100,lang_ext='en'):

		doc_batch = []
		doclen_batch = []
		sentlen_batch = []
		
		doc_dic = leidos_xml2dic(doc_file=doc_file)
		tokens_text = tokens_text_to_index(doc_dic["tokens_text"], lang_ext=lang_ext,
			dictionary=self.dictionary)
		doc_batch.append(tokens_text)

		doc_len = len(tokens_text)
		doclen_batch.append(doc_len)
		
		sentlen = []
		for i in range(0,len(tokens_text)):
			sentlen.append(len(tokens_text[i]))
		sentlen_batch.append(sentlen)
		sent_len = max(sentlen)

		for i in range(len(doc_batch)):
			doc_batch[i],sentlen_batch[i] = document_pad(doc_batch[i], 0, max_sent_len=sent_len, 
				doc_length=doc_len, sent_length=sentlen_batch[i])

		return np.array(doc_batch), np.array(doclen_batch), np.array(sentlen_batch)



















