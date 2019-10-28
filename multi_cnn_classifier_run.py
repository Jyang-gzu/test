import os
import codecs
from multi_cnn_classifier_utils import read_vocabs, build_vocabs, DataStruct, input_from_line
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from sklearn import metrics
from performance import performance_on_positives
from tensorflow.python.framework import graph_util

def classifier_build_vocabs(classify_config):
    if os.path.isfile(classify_config.vocab_path) and os.path.isfile(classify_config.label_path):
        word_to_id, tag_to_id = read_vocabs(classify_config.vocab_path, classify_config.label_path)
    else:
        word_to_id, tag_to_id = build_vocabs(classify_config.train_path, classify_config.vocab_path,
                                             classify_config.label_path, classify_config.vocab_size)
    return word_to_id, tag_to_id

def classifier_train(classify_config, classify_word_to_id, classify_tag_to_id, classify_model, boundary_config):

    for label in codecs.open("entityClassifier_vocabs/label.txt", 'r', 'utf-8').readlines():
        label = str(label).strip()
        if label == "yes":
            yes_id = 0
            no_id = 1
            break
        else:
            no_id = 0
            yes_id = 1
            break

    print("———————————————————————————————Training the classifier model—————————————————————————————————————")
    print("Loading classifier training and validation data_process", end='...')
    start_time = time.time()
    train_data = DataStruct(classify_config.train_path, classify_word_to_id, classify_tag_to_id, seq_length=classify_config.seq_length)
    val_data = DataStruct(classify_config.val_path, classify_word_to_id, classify_tag_to_id, seq_length=classify_config.seq_length)
    test_data = DataStruct(classify_config.test_path, classify_word_to_id, classify_tag_to_id, seq_length=classify_config.seq_length)
    time_dif = time.time() - start_time
    print("time usage:", timedelta(seconds=int(round(time_dif))))

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    best_val_f1 = 0.0  # 最佳验证集F1值
    classifier_log = codecs.open('log/classifier_log.txt', 'w')

    for epoch in range(classify_config.num_epochs):

        classifier_log.write(time.strftime("%Y-%m-%d %H:%M:%S") + '\n')
        print('Epoch: {0:>2}  ...'.format(epoch + 1))
        classifier_log.write('Epoch: {0:>2}  ...'.format(epoch + 1))
        train_true_cls, train_pred_cls = list(), list()

        for sentence, entity, label in train_data.batch_iter(classify_config.batch_size):
            # has_yes_label = False
            # y_prob, n_prob = [], []
            pred_cls = session.run(
                [classify_model.y_pred_cls, classify_model.probs, classify_model.optim],
                feed_dict={
                    classify_model.sentence_text: sentence,
                    classify_model.entity_text: entity,
                    classify_model.label: label,
                    classify_model.keep_prob: classify_config.dropout_keep_prob
                })
            # probs = pred_cls[1]
            # for prob in probs:
            #     if prob[yes_id] >= prob[no_id]:
            #         has_yes_label = True
            #         break
            # if has_yes_label:
            #     for prob in probs:
            #         y_prob.append(prob[yes_id])
            #     max_yes_prob = np.argmax(y_prob)
            #     for i in range(10):
            #         if i == max_yes_prob:
            #             train_pred_cls.append(yes_id)
            #         else:
            #             train_pred_cls.append(no_id)
            # else:
            #     for prob in probs:
            #         y_prob.append(1-prob[no_id])
            #     max_yes_prob = np.argmax(y_prob)
            #     for i in range(10):
            #         if i == max_yes_prob:
            #             train_pred_cls.append(yes_id)
            #         else:
            #             train_pred_cls.append(no_id)

            train_true_cls.extend(np.argmax(label, axis=1))
            train_pred_cls.extend(pred_cls[0])

        print('train report:')
        p, r, f = performance_on_positives(train_true_cls, train_pred_cls, boundary_config.train_data)
        #classifier_log.write('train report:' + '\n' + train_report)

        val_true_cls, val_pred_cls = list(), list()
        for sentence, entity, label in val_data.batch_iter(classify_config.batch_size):
            pred_cls = session.run(
                [classify_model.y_pred_cls, classify_model.probs],
                feed_dict={
                    classify_model.sentence_text: sentence,
                    classify_model.entity_text: entity,
                    classify_model.label: label,
                    classify_model.keep_prob: 1.0
                })
            val_true_cls.extend(np.argmax(label, axis=1))
            val_pred_cls.extend(pred_cls[0])
        print('val report:')
        # val_report = metrics.classification_report(val_true_cls, val_pred_cls, target_names=categories, digits=4)
        p_val, r_val, f_val = performance_on_positives(val_true_cls, val_pred_cls, boundary_config.val_data)

        if f_val > best_val_f1:
            print("new best dev f1")
            saver.save(sess=session, save_path=classify_config.checkpoint_path)
            best_val_f1 = f_val
            # 导入测试集

            test_true_lable, test_pred_label = list(), list()
            for sentence, entity, label in test_data.batch_iter(classify_config.batch_size):
                pred_cls = session.run(
                    [classify_model.y_pred_cls],
                    feed_dict={
                        classify_model.sentence_text: sentence,
                        classify_model.entity_text: entity,
                        classify_model.label: label,
                        classify_model.keep_prob: 1.0
                    }
                )
                test_true_lable.extend(np.argmax(label, axis=1))
                test_pred_label.extend(pred_cls[0])
            # test_report = metrics.classification_report(test_true_lable, test_pred_label, target_names=categories, digits=4)
            # p, r, f, support = metrics.precision_recall_fscore_support(test_true_lable, test_pred_label)
            print('test report:')
            p_test, r_test, f_test = performance_on_positives(test_true_lable, test_pred_label, boundary_config.test_data)
            # classifier_log.write('test report:' + '\n' + test_report + '\n')
            # classifier_log.write('test data for yes label and no label report:' + str(f[1]) + "," + str(f[0]) + '\n')
            # print(test_report)

    classifier_log.close()
    classify_config.save_config()
def classifier_predct(sess, classifier_input, classify_model, classify_word_to_id):
    real_entities, error_entity = list(), list()
    for input_js in classifier_input:
        sentences = input_js['left'] + '^' + input_js['entity'] + '$' + input_js['right']
        sentences_input = input_from_line(sentences, classify_word_to_id)
        entity = input_js['entity']
        entity_input = input_from_line(entity, classify_word_to_id)

        pred_cls = sess.run(
            [classify_model.y_pred_cls],
            feed_dict={
                classify_model.sentence_text: sentences_input[1],
                classify_model.entity_text: entity_input[1],
                classify_model.keep_prob: 1.0
            })
        if pred_cls[0][0] == 0:
            real_entities.append(entity)
        elif pred_cls[0][0] == 1:
            error_entity.append(entity)
        else:
            print("模型错误！")
    return real_entities, error_entity

