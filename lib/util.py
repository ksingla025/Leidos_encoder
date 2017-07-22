#!/usr/bin/python

''' Author : Karan Singla '''

#tensorflow imports
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib import rnn

def preprocess_text(text):
    
    text = text.strip()
    text = text.split()
    text = ' '.join(text)
    text = text.lower()
    return text

def gather_axis(params, indices, axis=-1):
    return tf.stack(tf.unstack(tf.gather(tf.unstack(params, axis=axis), indices)), axis=axis)


def map_fn_mult(fn, arrays, dtype=tf.float32):
    # assumes all arrays have same leading dim
    indices = tf.range(tf.shape(arrays[0])[0])
    out = tf.map_fn(lambda ii: fn(*[array[ii] for array in arrays]), indices, dtype=dtype)
    return out

def _attn_mul_fun(keys, query):

    return math_ops.reduce_sum(keys * query, [2])

def pad(l, content, width):
    
    l.extend([content] * (width - len(l)))
    return l

def document_pad(document, content, max_sent_len, doc_length, sent_length):

    # pad sentences to sen_len
    document = document[:doc_length]
    sent_length = sent_length[:doc_length]

    for i in range(0,len(document[:doc_length])):
        document[i] = pad(document[i][:max_sent_len], 0, max_sent_len)

    # pad sentences to the document
    if len(document) < doc_length:
        pad_sent = [content] * max_sent_len
        for i in range(0,(doc_length - len(document))):
            document.append(pad_sent)
            sent_length.append(0)

    return document, sent_length

def loss(x1, x2, y, margin = 0.0):
    ''' 
    calucaltes loss depending on cosine similarity and labels
    if label == 1:
        loss = 1 - cosine
    else:
        loss = max(0,cosine - margin)
    x1 : a 2D tensor ( batch_size, embed)
    x2 : a 2D tensor
    y : batch label tensor
    margin : margin for negtive samples loss
    '''

    #take dot product of x1,x2 : [batch_size,1]
    dot_products = tf.reduce_sum(tf.multiply(x1,x2),axis=1)

    # calulcate magnitude of two 1d tensors
    x1_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x1,x1),axis=1))
    x2_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x2,x2),axis=1))

    # calculate cosine distances between them
    cosine = dot_products / tf.multiply(x1_magnitudes,x2_magnitudes)

    # conver it into float and make it a row vector
    labels = tf.to_float(y)
    labels = tf.transpose(labels,[1,0])


    # you can try margin parameters, margin helps to set bound for mismatch cosine
    margin = tf.constant(margin)     

    # calculate number of match and mismatch pairs
    total_labels = tf.to_float(tf.shape(labels)[1])
    match_size = tf.reduce_sum(labels)
    mismatch_size = tf.subtract(total_labels,match_size)

    # loss culation for match and mismatch separately
    match_loss = 1 - cosine
    mismatch_loss = tf.maximum(0., tf.subtract(cosine, margin), 'mismatch_term')

    # combined loss for a batch
    loss_match = tf.reduce_sum(tf.multiply(labels, match_loss))
    loss_mismatch = tf.reduce_sum(tf.multiply((1-labels), mismatch_loss))

    # combined total loss
    # if label is 1, only match_loss will count, otherwise mismatch_loss
    loss = tf.add(tf.multiply(labels, match_loss), \
                  tf.multiply((1 - labels), mismatch_loss), 'loss_add')

    # take average for losses according to size
    loss_match_mean = tf.divide(loss_match,match_size)
    loss_mismatch_mean = tf.divide(loss_mismatch, mismatch_size)
    loss_mean = tf.divide(tf.reduce_sum(loss),total_labels)

    return loss_mean, loss_match_mean, loss_mismatch_mean
#    return loss_mean

def triplet_loss(x1, x2, x3, doc_len, margin = 0.0):
    '''
    x1, x2, x3 is a single document with aligned sentences
    x1, x2 are similar, whereas x3 is different from both
    '''
    # only take actual length of the document
    x1 = x1[:doc_len]
    x2 = x2[:doc_len]
    x3 = x3[:doc_len]

    # calulcate magnitude of two 1d tensors
    x1_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x1,x1),axis=1))
    x2_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x2,x2),axis=1))
    x3_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(x3,x3),axis=1))

    x1_magnitudes = tf.add(x1_magnitudes,0.1)
    x2_magnitudes = tf.add(x2_magnitudes,0.1)
    x3_magnitudes = tf.add(x3_magnitudes,0.1)

    #take dot product of x1,x2 : [batch_size,1]
    dot_products_x1x2 = tf.reduce_sum(tf.multiply(x1,x2),axis=1)
    dot_products_x1x3 = tf.reduce_sum(tf.multiply(x1,x3),axis=1)
    dot_products_x2x3 = tf.reduce_sum(tf.multiply(x2,x3),axis=1)

    dot_products_x1x2 = tf.add(dot_products_x1x2,0.0001)
    dot_products_x1x3 = tf.add(dot_products_x1x3,0.0001)
    dot_products_x2x3 = tf.add(dot_products_x1x3,0.0001)

    # calculate cosine distances between them
    cosine_x1x2 = dot_products_x1x2 / tf.multiply(x1_magnitudes,x2_magnitudes)
    cosine_x1x3 = dot_products_x1x3 / tf.multiply(x1_magnitudes,x3_magnitudes)
    cosine_x2x3 = dot_products_x2x3 / tf.multiply(x2_magnitudes,x3_magnitudes)

    print("cosine_x1x2", cosine_x1x2)
    print("cosine_x1x3", cosine_x1x3)
    # you can try margin parameters, margin helps to set bound for mismatch cosine
    margin = tf.constant(margin)

    # loss culation for match and mismatch separately
    match_loss = 1 - cosine_x1x2
    mismatch_loss_x1x3 = tf.maximum(0., tf.subtract(cosine_x1x3, margin), 'mismatch_term_x1x3')
    mismatch_loss_x2x3 = tf.maximum(0., tf.subtract(cosine_x2x3, margin), 'mismatch_term_x2x3')
    mismatch_loss = tf.add(mismatch_loss_x1x3, mismatch_loss_x2x3, 'mismatch_loss_add')

    doc_len = tf.to_float(doc_len)
    # combined loss for a batch
    loss_match = tf.reduce_sum(match_loss)
    loss_mismatch = tf.reduce_sum(mismatch_loss)

    loss = tf.add(loss_match, loss_mismatch)

    return loss, loss_match, loss_mismatch,cosine_x1x2, x1_magnitudes, x2_magnitudes, x3_magnitudes

def input_pipeline_skipgram(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_format_skipgram(filename_queue)
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

def read_format_skipgram(filename_queue):
    # file reader and value generator
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # default format of the file
    record_defaults = [[1], [1]]

    # get values in csv files
    col1, col2 = tf.decode_csv(value,record_defaults=record_defaults)

    # col1 is input, col2 is predicted label
    label = tf.stack([col2])
    return col1, label

def read_format_sentsim(filename_queue,max_length=50):

    # file reader and value generator
    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # default format of the file
    record_defaults = [[]] * ((max_length*2)+3)
    record_defaults[1].append(0)

    # extract all csv columns
    features = tf.decode_csv(value,record_defaults=record_defaults)

    # make different inputs for two sentences/examples
    example1 = features[:max_length]
    example2 = features[max_length:max_length*2]

    example1_len = features[-3]
    example2_len = features[-2]

    # extract label, whether they are similar/1 or not/0
    label = tf.to_float(features[-1])
    label = tf.stack([label])

    return example1, example2, example1_len, example2_len, label

def input_pipeline_sentsim(filenames, batch_size, num_epochs=None):

    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example1, example2, example1_len, example2_len, label = read_format_sentsim(filename_queue)
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
