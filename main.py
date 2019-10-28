import os
import codecs
import tensorflow as tf
from boundaryIdentify_model import Config, BiDirectionalRNNModel, create_model
from boundaryIdentify_utils import load_word2vec
from boundaryIdentify_run import build_vocabs, boundary_train, boundary_predict
from multi_cnn_classifier_model import TCNNConfig, multi_cnn_classifier
from multi_cnn_classifier_run import classifier_train, classifier_build_vocabs, classifier_predct
from boundary_combination import combination_model

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置GPU
    boundary_config = Config()
    boundary_word_to_id, boundary_tag_to_id = build_vocabs(boundary_config)
    # 修正字典大小
    boundary_config.vocab_size = len(boundary_word_to_id)
    boundary_config.tags = len(boundary_tag_to_id)
    boundary_model = BiDirectionalRNNModel(boundary_config)
    # 训练边界识别模型
    if boundary_config.train_model:
        boundary_train(boundary_config, boundary_word_to_id, boundary_tag_to_id, boundary_model)

    classify_config = TCNNConfig()
    classify_word_to_id, classify_tag_to_id = classifier_build_vocabs(classify_config)
    # 修正分类器的词典大小
    classify_config.vocab_size = len(classify_word_to_id)
    classify_model = multi_cnn_classifier(classify_config)
    # 训练多核CNN分类模型
    if classify_config.train_model:
        classifier_train(classify_config, classify_word_to_id, classify_tag_to_id, classify_model, boundary_config)
    # 调用模型（边界识别，边界组合，候选证据分类）
    if not boundary_config.train_model and not classify_config.train_model:
    # if not boundary_config.train_model:
    #     output_data1 = codecs.open("data/boundaryIdentify/fenci/result.utf8", 'a', 'utf-8')
        with codecs.open("data/boundaryIdentify/fenci/pku_test.utf8", 'r', 'utf-8') as f:
            for sentence in f.readlines():
                sentence = sentence.strip()
                # 边界识别
                tf_config = tf.ConfigProto()
                tf_config.gpu_options.allow_growth = True
                with tf.Session(config=tf_config) as sess:
                    boundary_model = create_model(sess, boundary_model, load_word2vec, boundary_config, boundary_word_to_id)
                    # boundary_model.saver.restore(sess, boundary_config.save_path)
                    # 把ckpt模型格式转成pb模型格式
                    # boundary_model.export_model(sess, boundary_model)
                    boundary_tags = boundary_predict(sess, sentence, boundary_word_to_id, boundary_tag_to_id, boundary_model)
                    print("边界识别的结果：", boundary_tags)
                    combination_model(boundary_tags)
        #             TEMP = boundary_tags[0]
        #             sentence = TEMP[0]
        #             tags = TEMP[1]
        #             for i, tag in enumerate(tags):
        #                 if tag == 0:
        #                     output_data1.write(' ' + sentence[i])
        #                     print(1111)
        #                 elif tag == 3:
        #                     output_data1.write(sentence[i])
        #                 elif tag == 1:
        #                     output_data1.write(sentence[i] + ' ')
        #                 else:  # tag == 'S'
        #                     output_data1.write(' ' + sentence[i] + ' ')
        #         output_data1.write("\r\n")
        # output_data1.close()


        # # 边界组合
        #     entities, combination_output = combination_model(boundary_tags)
        #     print("候选实体为：", entities)
        # # # 候选实体分类
        # with tf.Session(config=tf_config) as sess:
        #     classify_model.saver.restore(sess, classify_config.checkpoint_path)
        #     # 把ckpt模型格式转成pb模型格式
        #     # classify_model.export_model(sess, classify_model)
        #     real_entities, error_entity = classifier_predct(sess, combination_output, classify_model, classify_word_to_id)
        #     print("正确的实体：", real_entities)
        #     print("错误的实体：", error_entity)

