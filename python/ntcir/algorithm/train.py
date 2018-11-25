"""
trainin
"""
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-s: %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)
log.propagate = False

import json

from ntcir.algorithm import algorithm

import glob
import functools
from typing import Sequence
from pathlib import Path
from ntcir.algorithm import __version__
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import os
import time

run_modes = {'eval': tf.estimator.ModeKeys.EVAL,
             'train': tf.estimator.ModeKeys.TRAIN,
             'predict': tf.estimator.ModeKeys.PREDICT}

def init_flags():

    # global run options
    tf.flags.DEFINE_string("mode", "train", "the run mode")
    tf.flags.DEFINE_boolean("debug", False, "Whether to run with a tfdbg session (this won't work in an IDE)")

    # files
    tf.flags.DEFINE_string("vocab_file", "", "newline-delimited vocabulary file")
    tf.flags.DEFINE_string("char_file", "", "newline-delimited alphabet file")
    tf.flags.DEFINE_string("tags_file", "", "newline-delimited list of all tags")
    tf.flags.DEFINE_string("input_data_file", "", "newline-delimited sentences for classification")
    tf.flags.DEFINE_string("input_tags_file", "", "corresponding tags for the input data")
    tf.flags.DEFINE_string("eval_data_file", "", "newline-delimited sentences for evaluation")
    tf.flags.DEFINE_string("eval_tags_file", "", "corresponding tags for the eval data")
    tf.flags.DEFINE_string("glove_file", "", "compressed glove vectors (.npz)")
    tf.flags.DEFINE_string("empty_tag", "<O>", "tag signifying no relevant info (default: <O>)")

    # general files
    tf.flags.DEFINE_string("checkpoints", os.path.join(os.getcwd(), "checkpoints"), "checkpoint directory")
    tf.flags.DEFINE_string("modeldir", os.path.join(os.getcwd(), "models"), "models directory")
    tf.flags.DEFINE_string("name", str(int(time.time())), "run name")
    tf.flags.DEFINE_boolean("use_existing", False, "if present, selects the most recent model with this name to either resume training or evaluate")

    # model specific hyperparams
    tf.flags.DEFINE_integer("word_embedding_dim", 300, "Word embedding dim (default: 300)")
    tf.flags.DEFINE_integer("char_embedding_dim", 50, "Character embedding dim (default: 50)")
    tf.flags.DEFINE_boolean("use_gpu", False, "Whether to use CUDNN libs (default: false)")
    tf.flags.DEFINE_integer("num_oov_char_buckets", 5, "Number of out-of-vocab character buckets (default: 5)")
    tf.flags.DEFINE_integer("num_oov_word_buckets", 100, "Number of out-of-vocab word buckets (default: 100)")
    tf.flags.DEFINE_integer("conv_filters", 64, "Number of convolution filters (default: 64)")
    tf.flags.DEFINE_integer("conv_kernel_size", 2, "Convolution kernel size (default: 2)")
    tf.flags.DEFINE_integer("lstm_size", 100, "LSTM depth (default: 100)")
    tf.flags.DEFINE_integer("buffer", 15000, "not sure (default: 15000)")

    # general hyperparameters
    tf.flags.DEFINE_float("learning_rate", 1e-2, "Initial Learning Rate (default: 1e-2)")
    tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
    tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("max_iterations", -1, "Maximum iterations (default: -1 (no max))")
    tf.flags.DEFINE_integer("eval_steps", 10000, "Number of eval steps to run")
    tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_integer("lr_decay_steps", 100, "LR decays after this many steps")
    tf.flags.DEFINE_float("lr_decay_rate", 0.95, "LR exponential decay rate")
    tf.flags.DEFINE_integer("sequence_max_length", 256, "Sequence Max Length (default: 256)")

class RestoreHook(tf.train.SessionRunHook):
    def __init__(self, transfer_checkpoint_dir, checkpoint_dict):
        self.transfer_checkpoint_dir = transfer_checkpoint_dir
        self.checkpoint_dict = checkpoint_dict

    def after_create_session(self, session, coord=None):
        tf.train.init_from_checkpoint(self.transfer_checkpoint_dir, self.checkpoint_dict)


def create_run_hooks(flags) -> Sequence[tf.train.SessionRunHook]:
    """
    Creates session run hooks for summaries and checkpointing. Note that this method
    assumes the flags have been initializedd with the init_flags method in this module.
    :param flags: tf.flags.FLAGS
    :return: a summary saver and checkpoint saver run hook for a tf estimator
    """
    name = str(int(time.time())) + '-' + flags.name
    log_dir = os.path.join(flags.logdir, name)
    checkpoint_dir = os.path.join(flags.checkpoints, name)
    summary_hook = tf.train.SummarySaverHook(output_dir=log_dir, save_steps=1)
    checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir, save_steps=flags.evaluate_every)
    return [summary_hook, checkpoint_hook]


def parse_fn(line_words, line_tags):
    # Encode in Bytes for TF
    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Words and tags lengths don't match"

    # Chars
    chars = [[c.encode() for c in w] for w in line_words.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return ((words, len(words)), (chars, lengths)), tags

def generator_fn(words, tags):
    with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)

def make_parent_directory(path):
    parent = os.path.abspath(os.path.join(path, os.pardir))
    if not os.path.exists(parent):
        os.makedirs(parent)


def most_recent_checkpoint(checkpoints, name):
    # get the most recent dir matching this run name--start with wildcard [checkpoint]/*[name]
    wildcard_path = os.path.abspath(os.path.join(checkpoints, '*' + name))
    matching_paths = glob.glob(wildcard_path)
    matching_names = [Path(f).name for f in matching_paths]
    matching_names = sorted(matching_names, reverse=True)
    if len(matching_names) > 0:
        checkpoint_dir = os.path.join(checkpoints, matching_names[0])
        log.info("Using existing model at {}".format(checkpoint_dir))
        return checkpoint_dir
    else:
        raise NameError("No models matching name '{}' were found in the checkpoint directory '{}'".format(name, checkpoints))



if __name__ == '__main__':
    init_flags()
    flags = tf.flags.FLAGS
    tf.logging.set_verbosity(tf.logging.DEBUG)

    runmode = run_modes[flags.mode]
    # create checkpoint/logdir stuff
    name = str(int(time.time())) + '-' + flags.name
    # mkdirs
    checkpoint_dir = os.path.join(flags.checkpoints, name)

    if flags.use_existing or runmode != tf.estimator.ModeKeys.TRAIN:
        # get the most recent dir matching this run name--start with wildcard [checkpoint]/*[name]
        checkpoint_dir = most_recent_checkpoint(flags.checkpoints, flags.name)
    else:
        make_parent_directory(checkpoint_dir)

    hooks = []
    # add debug hooks if specified
    if flags.debug:
        hooks.append(tf_debug.LocalCLIDebugHook())
        hooks.append(tf.train.LoggingTensorHook(tensors=['IteratorGetNext:0', 'IteratorGetNext:1'], every_n_iter=1))

    with open(flags.tags_file) as t_f:
        tags = t_f.readlines()
    # Params
    params = {
        'vocab_file' : flags.vocab_file,
        'char_file' : flags.char_file,
        'word_embedding_dim' : flags.word_embedding_dim,
        'char_embedding_dim' : flags.char_embedding_dim,
        'tags' : tags,
        'tags_file' : flags.tags_file,
        'glove_location' : flags.glove_file,
        'dropout_pct' : flags.dropout_keep_prob,
        'use_gpu' : flags.use_gpu,
        'empty_tag' : flags.empty_tag,
        'num_oov_word_buckets' : flags.num_oov_word_buckets,
        'num_oov_char_buckets' : flags.num_oov_char_buckets,
        'epochs' : flags.num_epochs,
        'batch_size' : flags.batch_size,
        'buffer' : flags.buffer,
        'filters' : flags.conv_filters,
        'kernel_size' : flags.conv_kernel_size,
        'lstm_size' : flags.lstm_size,
    }

    model_fn = algorithm.model_fn

    def input_fn(words, tags, params=None, shuffle_and_repeat=False):
        params = params if params is not None else {}
        shapes = ((([None], ()),               # (words, nwords)
                   ([None, None], [None])),    # (chars, nchars)
                  [None])                      # tags
        types = (((tf.string, tf.int32),
                  (tf.string, tf.int32)),
                 tf.string)
        defaults = ((('<pad>', 0),
                     ('<pad>', 0)),
                    '<O>')
        dataset = tf.data.Dataset.from_generator(
            functools.partial(generator_fn, words, tags),
            output_shapes=shapes, output_types=types)

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20), shapes, defaults)
                   .prefetch(1))
        return dataset
    # Estimator, train and evaluate
    train_inpf = functools.partial(input_fn, flags.input_data_file, flags.input_tags_file,
                                   params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, flags.eval_data_file, flags.eval_tags_file)

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=cfg,
        params=params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=hooks)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120, hooks=hooks)
    if runmode == tf.estimator.ModeKeys.TRAIN:
        runconfig_file = os.path.join(flags.checkpoints, name + '.runconfig')
        with open(runconfig_file, 'w+') as runconfig:
            flagsdict = {k: v._value for k, v in flags.__flags.items()}
            flagsdict['__version__'] = __version__.local()
            json.dump(flagsdict, runconfig)
            log.info("Wrote run config to {}".format(os.path.abspath(runconfig.name)))
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    elif runmode == tf.estimator.ModeKeys.EVAL:
        hooks.append(tf.train.LoggingTensorHook(tensors=['eval_step:0'], every_n_secs=3))
        estimator.evaluate(input_fn=eval_inpf, steps=flags.eval_steps, hooks=hooks)

    # Write predictions to file
    def write_predictions(data_file_path, tag_file_path):
        Path(checkpoint_dir + '/score').mkdir(parents=True, exist_ok=True)
        with Path(checkpoint_dir + '/score/{}.preds.txt'.format(name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, data_file_path, tag_file_path)
            golds_gen = generator_fn(data_file_path, tag_file_path)
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), (_, _)), tags = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    # for giggles, i guess
    write_predictions(flags.eval_data_file, flags.eval_tags_file)