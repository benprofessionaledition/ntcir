import tensorflow as tf
import tensorflow.python.keras as keras

class SimpleBiLSTM(object):

    def __init__(self):
        self.model = keras.Sequential()
        self.forward = keras.layers.CuDNNLSTM