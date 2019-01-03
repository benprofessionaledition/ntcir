"""
Highway implementation
"""
import tensorflow as tf
import numpy as np

he_initializer = tf.contrib.keras.initializers.he_normal()


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

    char_seq_maxlen_ = params['char_seq_maxlen']
    word_seq_maxlen = params['word_seq_maxlen']

    (words, nwords), (chars, nchars) = features

    architecture = [2, 2, 2, 2]
    filter_multiplier = 32
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # doing this index table thing from tf.contrib because this way it can do the padding and lookup on the fly
    vocab_words = tf.contrib.lookup.index_table_from_file(vocab_file_, num_oov_buckets=num_oov_word_buckets_)
    vocab_chars = tf.contrib.lookup.index_table_from_file(char_vocab_file_, num_oov_buckets=num_oov_char_buckets_)
    num_tags = len(tags_)
    with open(char_vocab_file_, 'rb') as chars_in:
        num_chars = sum(1 for _ in chars_in) + num_oov_char_buckets_

    with tf.variable_scope("char_embeddings"):
        # pad this motherbitch
        char_ids = vocab_chars.lookup(chars)
        embedding_W = tf.Variable(tf.random_uniform([num_chars, char_embedding_dim_], -1.0, 1.0),
                                  name="embedding_W")
        embedded_characters = tf.nn.embedding_lookup(embedding_W, char_ids)
        embedded_characters_expanded = tf.transpose(embedded_characters, perm=[1,2,3,4,0])
    with tf.variable_scope("convolutions"):
        """ first reshape all this nonsense--needs to be [batch x maxlen x embedding x 1], right now it's like [batch x nwords x nchars x embedding x 1]"""
        character_slice = tf.slice(embedded_characters_expanded, )
        with tf.variable_scope("first_convolution"):
            filter_shape = [3, char_embedding_dim_, 1, filter_multiplier]
            w = tf.get_variable(name='W', shape=filter_shape,
                                initializer=he_initializer)
            conv = tf.nn.conv1d(embedded_characters_expanded, w, stride=char_embedding_dim_, padding="SAME")
            b = tf.get_variable(name='b', shape=[filter_multiplier],
                                initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(conv, b)
            first_conv = tf.nn.relu(out)

        def __convolutional_block(inputs, num_layers, num_filters, name, is_training):
            """
            A convolutional block which will be initialized with varying parameters a few times in the network
            :param inputs: the previous tensor
            :param num_layers: the number of layers
            :param num_filters: the number of filters
            :param name: a name for the tf name scope
            :param is_training: sets the 'is_training' parameter in the batch normalization
            :return:
            """
            with tf.variable_scope("conv_block_%s" % name):
                out = inputs
                for i in range(0, num_layers):
                    filter_shape = [3, 1, out.get_shape()[3], num_filters]
                    w = tf.get_variable(name='W_' + str(i), shape=filter_shape,
                                        initializer=he_initializer)
                    b = tf.get_variable(name='b_' + str(i), shape=[num_filters],
                                        initializer=tf.constant_initializer(0.0))
                    conv = tf.nn.conv2d(out, w, strides=[1, 1, 1, 1], padding="SAME")
                    conv = tf.nn.bias_add(conv, b)
                    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=is_training)
                    out = tf.nn.relu(batch_norm)
                return out
            # all convolutional blocks
        conv_block_1 = __convolutional_block(first_conv, num_layers=architecture[0], num_filters=filter_multiplier * 1, name='1',
                                             is_training=is_training)
        pool1 = tf.nn.max_pool(conv_block_1, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                               name="pool_1")

        conv_block_2 = __convolutional_block(pool1, num_layers=architecture[1], num_filters=filter_multiplier * 2, name='2',
                                             is_training=is_training)
        pool2 = tf.nn.max_pool(conv_block_2, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                               name="pool_2")

        conv_block_3 = __convolutional_block(pool2, num_layers=architecture[2], num_filters=filter_multiplier * 4, name='3',
                                             is_training=is_training)
        pool3 = tf.nn.max_pool(conv_block_3, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                               name="pool_3")

        conv_block_4 = __convolutional_block(pool3, num_layers=architecture[3], num_filters=filter_multiplier * 8, name='4',
                                             is_training=is_training)

        pool4 = tf.nn.max_pool(conv_block_4, ksize=[1, 3, 1, 1], strides=[1, 2, 1, 1], padding='SAME',
                               name="pool_4")
        shape = int(np.prod(pool4.get_shape()[1:]))
        conv_max_pool_out = tf.reshape(pool4, (-1, shape), name="conv_output")


    # an fc layer cause why not
    with tf.variable_scope("char_fc"):

        fc_out_dim = 1024 #idfk

        w = tf.get_variable('W_fc_aggregate', [conv_max_pool_out.get_shape()[1], fc_out_dim], initializer=he_initializer)
        b = tf.get_variable('b_fc_aggregate', [fc_out_dim], initializer=he_initializer)
        char_fc = tf.matmul(conv_max_pool_out, w) + b

    with tf.variable_scope("word_embeddings"):
        word_ids = vocab_words.lookup(words)
        glove = np.load(glove_location_)['embeddings']
        glove_stacked = np.vstack([glove, [[0.] * word_embedding_dim_]])
        glove_tensor = tf.Variable(glove_stacked, dtype=tf.float32, trainable=False)
        word_embeddings = tf.nn.embedding_lookup(glove_tensor, word_ids)

    with tf.variable_scope("word_char_embed_concat"):
        embeddings = tf.concat([word_embeddings, char_fc], axis=-1)
        embeddings = tf.nn.dropout(embeddings, keep_prob=dropout_pct_)

    # deprecating gpu support because it's Complicated(TM)
    if use_gpu_:
        RNNCellType = tf.contrib.cudnn_rnn.CudnnLSTM
        cellkwargs={'num_units': lstm_size_, 'num_layers': 2} # idfk?
    else:
        RNNCellType = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
        cellkwargs={'num_units': lstm_size_}

    with tf.variable_scope("bilstm"):
        time_major_embeddings = tf.transpose(embeddings, perm=[1, 0, 2])  # i guess
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        output_fw, _ = lstm_cell_fw(time_major_embeddings, dtype=tf.float32, sequence_length=nwords)
        output_bw, _ = lstm_cell_bw(time_major_embeddings, dtype=tf.float32, sequence_length=nwords)
        output = tf.concat((output_fw, output_bw), 2)
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