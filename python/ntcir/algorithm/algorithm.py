"""
Sequence Tagging Using CNN-BiLSTM-CRF
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import Embedding, Conv1D


class ConvBilstmCrf(object):

    def __init__(self, vocab_file, num_chars, tags, glove_location,
                 vocab_dim=300, char_dim=100, dropout_pct=0.5,
                 empty_tag='O', num_oov_buckets=1, epochs=25,
                 batch_size=20, buffer=15000, filters=50,
                 kernel_size=3, lstm_size=100):
        self.vocab_file = vocab_file
        self.num_chars = num_chars
        self.tags = tags
        self.glove_location = glove_location
        self.vocab_dim = vocab_dim
        self.char_dim = char_dim
        self.dropout_pct = dropout_pct
        self.empty_tag = empty_tag
        self.num_oov_buckets = num_oov_buckets # num out-of-vocab buckets for hashing crap
        self.epochs = epochs # training param
        self.batch_size = batch_size # training param
        self.buffer = buffer # training param
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_size = lstm_size

    def create_new(self):
        input_char = tf.keras.layers.Input()  # this will be padded
        input_words = tf.keras.layers.Input() # this won't

        char_embedding = Embedding(self.num_chars, self.char_dim)(input_char)
        char_conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, strides=2, padding='same', initializer='he_initializer')(char_embedding)

        # load glove
        with open(self.vocab_file, 'rb') as vocab_in:
            vocab_dict = { w.decode('utf-8') : i for i,w in enumerate(vocab_in) }
        glove = np.load(self.glove_location)['embeddings'] # this is hardcoded or something
        word_embedding = Embedding(len(vocab_dict), self.vocab_dim, weights=glove)(input_words)




