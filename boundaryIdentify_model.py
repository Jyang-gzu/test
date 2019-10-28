import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)


class Config:
    seg_dim = 20# Embedding size for segmentation, 0 if not used
    vocab_size = 20000
    embedding_dim = 100
    num_rnn_units = 100
    num_tags = 3
    tags = 0
    optimizer = 'adam'  # sgd adgrad
    max_to_keep = 1
    learning_rate = 0.005
    gradient_clip = 5
    keep_dropout = 0.5
    regularizer_scale = 0.5
    loss_type = 'crf_loss'  # or crf_loss cross_entropy_loss
    train_data = 'data/boundaryIdentify/fenci/train.utf8'
    val_data = 'data/boundaryIdentify/fenci/dev.utf8'
    test_data = 'data/boundaryIdentify/fenci/test.utf8'
    activate_test = True
    train_model = True
    vocab_file = 'boundaryIdentify_vocabs/vocab-p.txt'
    label_file = 'boundaryIdentify_vocabs/label-p.txt'
    batch_size = 20
    num_epochs = 30
    # save_path = 'boundaryIdentify_checkpoints/ckp-p/best_val_model'
    save_path = 'boundaryIdentify_checkpoints'


    def __init__(self):
        boundaryIdentify_ckp_dir = "boundaryIdentify_checkpoints"
        boundaryIdentify_vocab_dir = "boundaryIdentify_vocabs"
        if not os.path.isdir(boundaryIdentify_ckp_dir):
            os.mkdir(boundaryIdentify_ckp_dir)
        if not os.path.isdir(boundaryIdentify_vocab_dir):
            os.mkdir(boundaryIdentify_vocab_dir)

    def keys(self):
        # return 'train_data', 'val_data', 'test_data'
        return [i for i in dir(self) if i[-2:] != '__' and not callable(getattr(self, i))]

    def __getitem__(self, item):
        return getattr(self, item)

class BiDirectionalRNNModel(object):

    def __init__(self, config):
        self.config = config
        self.num_heads = 8
        self.num_units = 200
        self.num_segs = 4
        self.initializer = tf.contrib.layers.xavier_initializer()
        self.regularizer = tf.contrib.layers.l2_regularizer(config.regularizer_scale)

        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='char_inputs')
        self.char_labels = tf.placeholder(dtype=tf.int32, shape=[None, None], name='char_labels')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")

        self.real_lengths = tf.cast(tf.reduce_sum(tf.sign(tf.abs(self.char_inputs)), 1), tf.int32)
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        # 在第一块cpu上运行
        self.embedding = []
        with tf.device('/cpu:0'):
            self.lookup_table = tf.get_variable(
                name="embedding",
                shape=[self.config.vocab_size, self.config.embedding_dim])
            self.embedding.append(tf.nn.embedding_lookup(self.lookup_table, self.char_inputs))
            if self.config.seg_dim:
                self.seg_lookup = tf.get_variable(
                    name="seg_embedding",
                    shape=[self.num_segs, self.config.seg_dim],
                    initializer=self.initializer
                )
                self.embedding.append(tf.nn.embedding_lookup(self.seg_lookup, self.seg_inputs))
        embedd = tf.concat(self.embedding, axis=-1)
        rnn_inputs = tf.nn.dropout(embedd, self.keep_prob)

        with tf.name_scope('lstm'):
            rnn_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    rnn_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
                        num_units=self.config.num_rnn_units,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True
                    )
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=rnn_cell["forward"],
                cell_bw=rnn_cell["backward"],
                inputs=rnn_inputs,
                dtype=tf.float32,
                sequence_length=self.real_lengths)
            lstm_outputs = tf.concat(outputs, axis=2)
            # att_outputs = self.self_attention(lstm_outputs)
        with tf.variable_scope("hidden_layer"):
            W = tf.get_variable(
                name="W",
                shape=[self.config.num_rnn_units*2, self.config.num_rnn_units],
                dtype=tf.float32,
                initializer=self.initializer,
                regularizer=self.regularizer)
            b = tf.get_variable(
                name="b",
                shape=[self.config.num_rnn_units],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                regularizer=self.regularizer
            )
            output = tf.reshape(lstm_outputs, shape=[-1, self.config.num_rnn_units * 2])
            fc = tf.nn.xw_plus_b(output, W, b)
            # fc_dropout = tf.contrib.layers.dropout(fc, self.keep_prob)
            hidden = tf.nn.tanh(fc)
            # pred = tf.nn.xw_plus_b(output, W, b)
        # self.char_outputs = tf.reshape(pred, [-1, self.num_steps, self.config.num_tags])

        with tf.variable_scope("logits"):
            W = tf.get_variable(
                name="W",
                shape=[self.config.num_rnn_units, self.config.num_tags],
                dtype=tf.float32,
                initializer=self.initializer,
                regularizer=self.regularizer
            )
            b = tf.get_variable(
                name="b",
                shape=[self.config.num_tags],
                dtype=tf.float32,
                initializer=tf.zeros_initializer(),
                regularizer=self.regularizer
            )
            pred = tf.nn.xw_plus_b(hidden, W, b)
            self.char_outputs = tf.reshape(pred, [-1, self.num_steps, self.config.num_tags])
            # tf.nn.softmax(self.char_outputs, dim=-1, name='probability')
            # tf.argmax(self.char_outputs, axis=-1, name='classes')

        with tf.variable_scope("optimizer"):
            if self.config.optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.config.learning_rate)
            elif self.config.optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.config.learning_rate)
            elif self.config.optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.config.learning_rate)
            else:
                raise KeyError('the config.optimizer wrong.')
        if self.config.loss_type == 'cross_entropy_loss':
            with tf.name_scope("cross_entropy_loss"):
                # 计算交叉熵损失函数
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.char_outputs,
                                                                               labels=self.char_labels)
                self.loss = tf.reduce_mean(cross_entropy)

            with tf.variable_scope("optimizer"):
                self.train_op = self.opt.minimize(self.loss)

        elif self.config.loss_type == 'crf_loss':
            # CRF层
            with tf.variable_scope("crf_loss"):
                small = -1000.0
                start_logits = tf.concat(
                    [
                        small * tf.ones(shape=[self.batch_size, 1, self.config.num_tags]),
                        tf.zeros(shape=[self.batch_size, 1, 1])
                    ],
                    axis=-1)
                pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
                logits = tf.concat([self.char_outputs, pad_logits], axis=-1)
                logits = tf.concat([start_logits, logits], axis=1)
                targets = tf.concat(
                    [
                        tf.cast(self.config.num_tags * tf.ones([self.batch_size, 1]), tf.int32),
                        self.char_labels
                    ],
                    axis=-1)
                self.trans = tf.get_variable(
                    name="transitions",
                    shape=[self.config.num_tags + 1, self.config.num_tags + 1],
                    dtype=tf.float32,
                    initializer=self.initializer)
                log_likelihood, self.trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits,
                    tag_indices=targets,
                    transition_params=self.trans,
                    sequence_lengths=self.real_lengths + 1)
                self.loss = tf.reduce_mean(-log_likelihood)

            with tf.variable_scope("optimizer"):
                # apply grad clip to avoid gradient explosion
                grads_vars = self.opt.compute_gradients(self.loss)
                capped_grads_vars = [
                    [tf.clip_by_value(g, -self.config.gradient_clip, self.config.gradient_clip), v]
                    for g, v in grads_vars]
                self.train_op = self.opt.apply_gradients(capped_grads_vars)
        else:
            raise KeyError('the config.loss_type wrong.')

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    
    def normalize(self, inputs, epsilon=1e-8, scope="ln", reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            inputs_shape = inputs.get_shape()
            params_shape = inputs_shape[-1:]
            mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
            beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)
            gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
            normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
            outputs = gamma * normalized + beta
        return outputs

    # attention
    def self_attention(self, keys, scope='multihead_attention', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            Q = tf.nn.relu(
                tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            K = tf.nn.relu(
                tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            V = tf.nn.relu(
                tf.layers.dense(keys, self.num_units, kernel_initializer=tf.contrib.layers.xavier_initializer()))
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [self.num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(keys)[1], 1])
            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            query_masks = tf.tile(query_masks, [self.num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)
            outputs += keys
            outputs = self.normalize(outputs)
        return outputs

    # 转pb
    def export_model(self, session, m):

        # 只需要修改这一段，定义输入输出，其他保持默认即可
        model_signature = signature_def_utils.build_signature_def(
            inputs={"char_inputs": utils.build_tensor_info(m.char_inputs)},
            outputs={
                "chat_outputs": utils.build_tensor_info(m.char_outputs)},

            method_name=signature_constants.PREDICT_METHOD_NAME)

        export_path = "boundary_model/pb"
        if os.path.exists(export_path):
            os.system("rm -rf " + export_path)
        print("Export the model to {}".format(export_path))

        try:
            legacy_init_op = tf.group(
                tf.tables_initializer(), name='legacy_init_op')
            builder = saved_model_builder.SavedModelBuilder(export_path)
            builder.add_meta_graph_and_variables(
                session, [tag_constants.SERVING],
                clear_devices=True,
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        model_signature,
                },
                legacy_init_op=legacy_init_op)

            builder.save()
        except Exception as e:
            print("Fail to export saved model, exception: {}".format(e))

    def train_batch(self, sess, batch_data):
        loss, _ = sess.run(
            [self.loss, self.train_op],
            {
                self.char_inputs: batch_data[1],
                self.seg_inputs: batch_data[2],
                self.char_labels: batch_data[3],
                self.keep_prob: self.config.keep_dropout
            }
        )
        return loss

    def run_batch(self, train, sess, batch_data):
        if train:
            lengths, outputs = sess.run(
                [self.real_lengths, self.char_outputs],
                {
                    self.char_inputs: batch_data[1],
                    self.seg_inputs: batch_data[2],
                    self.keep_prob: 1.0
                }
            )
        else:
            lengths, outputs = sess.run(
                [self.real_lengths, self.char_outputs],
                {
                    self.char_inputs: batch_data[1],
                    self.seg_inputs: batch_data[2],
                    self.keep_prob: 1.0
                }
            )
        return lengths, outputs


    def decode(self, logits, lengths, matrix):
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.config.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            if matrix is None:
                paths.append(np.argmax(score, axis=-1))
            else:
                pad = small * np.ones([length, 1])
                logits = np.concatenate([score, pad], axis=1)
                logits = np.concatenate([start, logits], axis=0)
                path, _ = tf.contrib.crf.viterbi_decode(logits, matrix)
                paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager):
        if self.config.loss_type == 'cross_entropy_loss':
            trans = None
        elif self.config.loss_type == 'crf_loss':
            trans = self.trans.eval()
        else:
            raise KeyError('the config.loss_type wrong.')
        golds, preds = [], []
        for i, batch in enumerate(data_manager.iter_batch()):
            strings = batch[0]
            tags = batch[-1]
            lengths, scores = self.run_batch(True, sess, batch)
            batch_paths = self.decode(scores, lengths, trans)

            for i in range(len(strings)):
                golds.extend([int(x) for x in tags[i][:lengths[i]]])
                preds.extend([int(x) for x in batch_paths[i][:lengths[i]]])
        return golds, preds

    def predict(self, sess, inputs, id_to_tag):
        if self.config.loss_type == 'cross_entropy_loss':
            trans = None
        elif self.config.loss_type == 'crf_loss':
            trans = self.trans.eval()
        else:
            raise KeyError('the config.loss_type wrong.')
        lengths, scores = self.run_batch(False, sess, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        return batch_paths[0]

def create_model(session, boundary_model, load_vec, config, id_to_char):

    ckpt = tf.train.get_checkpoint_state(config.save_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        #logger.info()输出到kong控制台
        # print("Reading model parameters from %s" % ckpt)
        print("加载模型")
        boundary_model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())  # 参数初始化
        #if config["pre_emb"]:
        emb_weights = session.run(boundary_model.lookup_table.read_value())
        emb_weights = load_vec("wiki_100.utf8", id_to_char, config.embedding_dim, emb_weights)
        session.run(boundary_model.lookup_table.assign(emb_weights))
        print("Load pre-trained embedding.")
    return boundary_model
