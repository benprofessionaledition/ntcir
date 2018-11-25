"""
Sequence Tagging Using CNN-BiLSTM-CRF
"""
from functools import reduce

import numpy as np
import tensorflow as tf


def masked_1dconv_max(t, weights, filters, kernel_size):
    """Applies 1d convolution and a masked max-pooling

    Parameters
    ----------
    t : tf.Tensor
        A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    weights : tf.Tensor of tf.bool
        A Tensor of shape [d1, d2, dn-1]
    filters : int
        number of filters
    kernel_size : int
        kernel size for the temporal convolution

    Returns
    -------
    tf.Tensor
        A tensor of shape [d1, d2, dn-1, filters]

    """
    # todo look closer at this
    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x * y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims - 2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max


def model_fn(features: tf.Tensor, labels: tf.Tensor, mode: tf.estimator.ModeKeys, params: dict) -> tf.estimator.EstimatorSpec:
    """
    Constructs some bizarre bastardization of a BiLSTM-CharCNN-CRF that I'm making up as I go

    :return: a model function suitable for use with the tf Estimator API
    """

    # using the underscore suffix to denote parameters defined externally
    vocab_file_ = params['vocab_file']
    char_vocab_file_ = params['char_file']
    word_embedding_dim_ = params['word_embedding_dim']
    char_embedding_dim_ = params['char_embedding_dim']
    tags_ = params['tags']
    tags_file_ = params['tags_file']
    glove_location_ = params['glove_location']
    dropout_pct_ = params['dropout_pct']
    use_gpu_ = params['use_gpu']
    num_oov_word_buckets_ = params['num_oov_word_buckets']  # num out-of-vocab buckets for hashing crap
    num_oov_char_buckets_ = params['num_oov_char_buckets']
    filters_ = params['filters']
    kernel_size_ = params['kernel_size']
    lstm_size_ = params['lstm_size']

    (words, nwords), (chars, nchars) = features

    # doing this index table thing from tf.contrib because this way it can do the padding and lookup on the fly
    vocab_words = tf.contrib.lookup.index_table_from_file(vocab_file_, num_oov_buckets=num_oov_word_buckets_)
    vocab_chars = tf.contrib.lookup.index_table_from_file(char_vocab_file_, num_oov_buckets=num_oov_char_buckets_)
    num_tags = len(tags_)
    with open(char_vocab_file_, 'rb') as chars_in:
        num_chars = sum(1 for _ in chars_in) + num_oov_char_buckets_

    with tf.variable_scope("char_embeddings"):
        char_ids = vocab_chars.lookup(chars)
        char_var = tf.get_variable('chars', [num_chars + 1, char_embedding_dim_], tf.float32)
        char_embeddings = tf.nn.embedding_lookup(char_var, char_ids)  # !

    with tf.variable_scope("char_convolution"):
        weights = tf.sequence_mask(nchars)
        # masked convolution, so we can get rid of the stuff we don't care about
        char_conv = masked_1dconv_max(char_embeddings, weights, filters_, kernel_size_)  # !

    with tf.variable_scope("word_embeddings"):
        word_ids = vocab_words.lookup(words)
        glove = np.load(glove_location_)['embeddings']
        glove_stacked = np.vstack([glove, [[0.] * word_embedding_dim_]])
        glove_tensor = tf.Variable(glove_stacked, dtype=tf.float32, trainable=False)
        word_embeddings = tf.nn.embedding_lookup(glove_tensor, word_ids)

    with tf.variable_scope("word_char_embed_concat"):
        embeddings = tf.concat([word_embeddings, char_conv], axis=-1)
        embeddings = tf.nn.dropout(embeddings, keep_prob=dropout_pct_)

    if use_gpu_:
        RNNCellType = tf.contrib.cudnn_rnn.CudnnLSTM
    else:
        RNNCellType = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell

    with tf.variable_scope("bilstm"):
        time_major_embeddings = tf.transpose(embeddings, perm=[1, 0, 2])  # i guess
        lstm_fwd = RNNCellType(num_units=lstm_size_)
        lstm_bkwd = RNNCellType(num_units=lstm_size_)
        (output_fwd, output_bkwd), (_, _) = tf.nn.bidirectional_dynamic_rnn(lstm_fwd, lstm_bkwd, time_major_embeddings, sequence_length=nwords, time_major=True, dtype=tf.float32)
        output = tf.concat((output_fwd, output_bkwd), 2)
        output = tf.transpose(output, perm=[1, 0, 2])

    with tf.variable_scope("crf"):
        logits = tf.layers.dense(output, num_tags)
        crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(tags_file_)
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(tags_file_)
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': tf.metrics.precision(tags, pred_ids, weights),
            'recall': tf.metrics.recall(tags, pred_ids, weights)
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)