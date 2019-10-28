import os
import numpy as np
import time
import tensorflow as tf
from boundaryIdentify_utils import create_vocabs, load_vocabs, DataSet, input_from_line, load_sentences, prepare_dataset
from boundaryIdentify_model import create_model
from boundaryIdentify_utils import load_word2vec
from sklearn import metrics

def build_vocabs(boundary_config):
    if os.path.isfile(boundary_config.vocab_file) and os.path.isfile(boundary_config.label_file):
        word_to_id, tag_to_id = load_vocabs(boundary_config.vocab_file, boundary_config.label_file)
    else:
        word_to_id, tag_to_id = create_vocabs(boundary_config.train_data, boundary_config.vocab_file,
                                              boundary_config.label_file, boundary_config.vocab_size)
    return word_to_id, tag_to_id

def boundary_train(boundary_config, word_to_id, tag_to_id, boundary_model):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # 构造数据
    train_seg = load_sentences(boundary_config.train_data)
    val_seg = load_sentences(boundary_config.val_data)
    test_seg = load_sentences(boundary_config.test_data)

    train_data = prepare_dataset(train_seg, word_to_id, tag_to_id, True)
    val_data = prepare_dataset(val_seg, word_to_id, tag_to_id, True)
    test_data = prepare_dataset(test_seg, word_to_id, tag_to_id, True)

    train_manager = DataSet(train_data, 20) # 20 is train batch_size
    val_manager = DataSet(val_data, 100)
    test_manager = DataSet(test_data, 100)
    # 获得标签分类
    labels_sort = [k[0] for k in sorted(tag_to_id.items(), key=lambda k: k[1])]
    steps_check = 100
    best_dev_f1 = 0.0

    with tf.Session(config=tf_config) as sess:
        boundary_model = create_model(sess, boundary_model, load_word2vec, boundary_config, word_to_id)
        print("继续训练")
        sess.run(tf.global_variables_initializer())
        print("———————————————————————————————Training the boundary identify model—————————————————————————————————————")
        loss = list()
        for epoch in range(boundary_config.num_epochs):
            for step, batch in enumerate(train_manager.iter_batch(shuffle=True)):
                batch_loss = boundary_model.train_batch(sess, batch)
                loss.append(batch_loss)
                if step % steps_check == 0 or step+1 == train_manager.num_batch:
                    print("iteration:{} step:{}/{}, loss:{:>9.6f}".format(
                        epoch+1, step+1, train_manager.num_batch, np.mean(loss)))
                    loss = []
            val_truth_ys, val_pred_ys = boundary_model.evaluate(sess, val_manager)
            p, r, f1, s = metrics.precision_recall_fscore_support(val_truth_ys, val_pred_ys)
            # print(s[0], "\t", p[0], "\n", r[0], "\n", f1[0], "\n")
            # print(s[1], "\t", p[1], "\n", r[1], "\n", f1[1], "\n")
            # print(s[2], "\t", p[2], "\n", r[2], "\n", f1[2], "\n")
            curr_dev_f1 = np.average(f1, weights=s)
            curr_dev_f1_tag = ''
            if curr_dev_f1 > best_dev_f1:
                curr_dev_f1_tag = ' -*- '
                boundary_model.saver.save(sess, os.path.join(boundary_config.save_path, "best_val"))
                best_dev_f1 = curr_dev_f1

            train_truth_ys, train_pred_ys = boundary_model.evaluate(sess, train_manager)

            train_report = "train PRF report:\n" + metrics.classification_report(
                train_truth_ys, train_pred_ys, target_names=labels_sort, digits=4)
            val_report = "val PRF report:" + curr_dev_f1_tag + '\n' + metrics.classification_report(
                val_truth_ys, val_pred_ys, target_names=labels_sort, digits=4)
            print(train_report)
            print(val_report)

            with open('log/boundary_identify_log.txt', 'a') as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + '\n')
                f.write(train_report)
                f.write(val_report)
                if boundary_config.activate_test:
                    test_truth_ys, test_pred_ys = boundary_model.evaluate(sess, test_manager)
                    test_report = "test PRF report:\n" + metrics.classification_report(
                        test_truth_ys, test_pred_ys, target_names=labels_sort, digits=4)
                    print(test_report)
                    f.write(test_report)
                f.write('\n')


def boundary_predict(sess, sentence, boundary_word_to_id, boundary_tag_to_id, boundary_model):

    all_tags = []
    temp_str = sentence.replace("，", "，|").replace("；", "；|").replace("。", "。|").replace("？", "？|").replace("！",  "！|").replace("：", "：|").replace(" ", "|").replace(",", ",|")
    contents = temp_str.split("|")

    for content in contents:
        if content != "":
            sentence_tags = []
            tags = boundary_model.predict(sess, input_from_line(content, boundary_word_to_id), boundary_tag_to_id)
            sentence_tags.append(content)
            sentence_tags.append(tags)
            all_tags.append(sentence_tags)

    return all_tags

