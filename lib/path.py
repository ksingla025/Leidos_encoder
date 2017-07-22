#!/usr/bin/python

from subprocess import call

DATA = "./data/"
DATA_BI = DATA+"parallel/"
DATA_MONO = DATA+"mono/"

DATA_TASK = DATA+"task/"
DATA_PROCESSED = DATA +"processed/"

DATA_TED_CLEAN = DATA+"ted-cldc/"

DATA_CLS_CLEAN = DATA+"cls-acl10-processed/"

DATA_CLDC = DATA+"cldc/"

DATA_MONO_CLEAN = DATA_PROCESSED + "mono_clean/"
call("mkdir -p "+DATA_MONO_CLEAN, shell=True)

DATA_BI_CLEAN = DATA_PROCESSED + "bi_clean/"
call("mkdir -p "+DATA_BI_CLEAN, shell=True)

DATA_ID = DATA_PROCESSED + "word2id/"
call("mkdir -p "+DATA_ID, shell=True)

DATA_BATCH = DATA_PROCESSED + "batch/"
call("mkdir -p "+DATA_BATCH, shell=True)

LOGS_PATH = './logs_doc2vec/'
call("mkdir -p "+LOGS_PATH, shell=True)

GRAPH_FILES = './graphs_sentsim'
call("mkdir -p "+GRAPH_FILES, shell=True)

MODEL_PATH = './models/'
call("mkdir -p "+MODEL_PATH, shell=True)

# leidos data paths
DATA_LEIDOS = DATA+"leidos/"
DATA_LEIDOS_ES_TEST = DATA_LEIDOS+"pretrans_tokenized_es.json"
DATA_LEIDOS_FR_TEST = DATA_LEIDOS+"pretrans_tokenized_fr.json"
DATA_LEIDOS_EN_TRAIN = DATA_LEIDOS+"tokenized_3_en_train.json"
DATA_LEIDOS_EN_TEST = DATA_LEIDOS+"tokenized_3_en_test.json"

# ldc data paths
DATA_LDC_SF = DATA+"ldc_sf/"

#yelp data path
DATA_YELP = DATA+"yelp/"