import os
import tensorflow as tf
import json
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, signature_def_utils, tag_constants, utils)

class TCNNConfig(object):
    """TCnn参数配置"""
    seq_length = None
    max_to_keep = 1
    num_classes = 2  # 类别数
    vocab_size = 5000  # 词汇表默认大小，需根据实际情况更改
    embedding_dim = 64  # 词向量维度
    kernel_size = [2, 3, 4, 5]  # 每个卷积核的尺寸
    num_filters = 64  # 每个卷积核的数目
    hidden_dim = 128  # 全连接层神经元
    learning_rate = 0.01  # 学习率
    dropout_keep_prob = 0.5  # dropout保留比例
    batch_size = 64  # 每批训练大小
    num_epochs = 40  # 总迭代轮次

    train_model = False
    train_path = 'data/entityClassifier/evidence/BE/train.json'
    test_path = 'data/entityClassifier/evidence/BE/test.json'
    val_path = 'data/entityClassifier/evidence/BE/val.json'

    vocabs_dir = 'entityClassifier_vocabs'
    vocab_path = os.path.join(vocabs_dir, 'vocab.txt')
    label_path = os.path.join(vocabs_dir, 'label.txt')
    config_path = os.path.join(vocabs_dir, 'cnn_entity_config.json')

    # checkpoint_dir = 'entityClassifier_checkpoints'  # 最佳验证结果保存路径
    checkpoint_dir = 'entityClassifier_checkpoints'  # 最佳验证结果保存路径
    checkpoint_path = os.path.join(checkpoint_dir, 'cnn_entity_classifier')

    def __init__(self):
        if not os.path.isdir(self.vocabs_dir):
            os.mkdir(self.vocabs_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def save_config(self, path=config_path):
        config = {
            'num_classes': self.num_classes,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.learning_rate,
            'dropout_keep_prob': self.dropout_keep_prob,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs
        }
        with open(path, 'w') as f:
            f.write(json.dumps(config, ensure_ascii=False, indent=4))

class multi_cnn_classifier(object):
    def __init__(self, config):
        self.config = config

        self.entity_text = tf.placeholder(
            tf.int32, [None, None], name='entity_text')
        self.sentence_text = tf.placeholder(
            tf.int32, [None, None], name='sentence_text')
        self.label = tf.placeholder(
            tf.float32, [None, self.config.num_classes], name='label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            entity_embedding = tf.get_variable(
                'entity_embedding', [self.config.vocab_size, self.config.embedding_dim])
            entity_inputs = tf.nn.embedding_lookup(entity_embedding, self.entity_text)

            sentence_embedding = tf.get_variable(
                'sentence_embedding', [self.config.vocab_size, self.config.embedding_dim])
            sentence_inputs = tf.nn.embedding_lookup(sentence_embedding, self.sentence_text)
        # 卷积
        with tf.name_scope("cnn"):
            gmps = list()
            for kernel_size in self.config.kernel_size:
                conv = tf.layers.conv1d(entity_inputs, self.config.num_filters,
                                        kernel_size, padding='same')
                gmps.append(tf.reduce_max(conv, reduction_indices=[1]))

                conv1 = tf.layers.conv1d(sentence_inputs, self.config.num_filters,
                                        kernel_size, padding='same')
                gmps.append(tf.reduce_max(conv1, reduction_indices=[1]))

            gmp = tf.concat(gmps, axis=-1)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.probs = tf.nn.softmax(self.logits, name='probability')

            self.y_pred_cls = tf.argmax(self.probs, 1, name='y_pred_cls')  # 预测类别
        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.label, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.max_to_keep)

    def export_model(self, session, m):
        # 只需要修改这一段，定义输入输出，其他保持默认即可
        model_signature = signature_def_utils.build_signature_def(
            inputs={"entity_text_input": utils.build_tensor_info(m.entity_text), "sentence_text_input": utils.build_tensor_info(m.sentence_text)},
            outputs={
                "output": utils.build_tensor_info(m.y_pred_cls)},

            method_name=signature_constants.PREDICT_METHOD_NAME)

        export_path = "classifier_model/pb"
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

