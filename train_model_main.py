import codecs
import re
import os
import numpy as np
from boundaryIdentify_model import Config, BiDirectionalRNNModel, create_model
from boundaryIdentify_utils import load_word2vec
from boundaryIdentify_run import build_vocabs, boundary_train
from boundaryIdentify_utils import input_from_line
from multi_cnn_classifier_model import TCNNConfig, multi_cnn_classifier
from multi_cnn_classifier_run import classifier_train, classifier_build_vocabs
import json
import tensorflow as tf

def character_tagging(input_file):
    path = os.listdir(input_file)
    content_dic = {}
    doc_name = []
    for doc in path:
        doc_path =input_file + doc
        # print(doc_path)
        with codecs.open(doc_path, "r", "utf-8") as f:
            lines = f.readlines()
        content = ""
        for line in lines:
            if line[1:5] != "List" and line[1:4] != 'Doc' and line[1:6] != "/List" and line[1:5] != '/Doc':
                content += line.rstrip()
        c = content.replace(' ', '')
        p = re.findall(r"<P>(.+?)</P>", c)  # p表示证据块

        for i, p_content in enumerate(p):
            doc_name.append(doc_path+str(i))
            content_dic[doc_path+str(i)] = p_content
    files = os.listdir("check")
    for filename in files:
        # print(filename)
        with open("check/" + filename, 'r', encoding='UTF-8') as f:
            for j, line in enumerate(f.readlines()):
                line = eval(line)
                evidence = line["evidence"]
                evidence_name = str(line["file_name"]) + str(line["ah"]) + str(j)
                p_content = re.findall(r"<P>(.+?)</P>", evidence)
                doc_name.append(evidence_name)
                content_dic[evidence_name] = p_content[0].replace(' ', '')
    print(len(doc_name))
    print(len(content_dic))
    return content_dic, doc_name

def do_random(name):

    np.random.shuffle(name)
    train_size = int(len(name)*0.6)
    print(train_size)
    dev_size = int(len(name)*0.2)
    print(dev_size)
    train = name[:train_size]
    dev = name[train_size:dev_size + train_size]
    test = name[dev_size + train_size:]
    # test_dev = name[train_size:]
    # dev_size = int(len(test_dev) * 0.5)
    # dev=test_dev[:dev_size]
    # test=test_dev[dev_size:]
    return train, dev, test

def do_tag(name, dic, path):
    #提取证据关键字，
    output_data = codecs.open(path, 'w', 'utf-8')
    for n in name:
        # print(path)
        s = []
        pi_str = dic[str(n)].replace(",", "，")
        D = re.findall(r"<D>(.+?)</D>", pi_str)  # D表示证据
        # D = re.findall(r"<H>(.+?)</H>", pi_str)  # H表示证据中心词
        # sentences = re.split(u'[<][D][>]', pi_str)
        sentences = re.split(u'[<][D][>]', pi_str)# zheli
        for seg in sentences:
            ss = re.split(u'[<][/][D][>]', seg)# zheli
            for i in ss:
                s.append(i)
        # print(s)
        isO = True
        for word_list in s:#所有字符合集
            for D_list in D:
                if word_list == D_list:
                    isO = False
                    break
            if isO:
                word = re.split(u'[\n]', word_list.replace("<H>", "").replace("</H>", "").replace("<M>", "").replace("</M>", ""))
                for char in word:
                    char_list = list(char)
                    for tag in char_list:
                        if tag == "<" or tag == ">" or tag == "/" or tag == "D" or tag == "H" or tag == "P":
                            del tag
                            continue
                        if tag != "<" and tag != ">" and tag != "/" and tag != "D" and tag != "H" and tag != "P" and tag != "M":
                            output_data.write(tag + " O")
                            output_data.write("\r\n")
                            if tag == '，' or tag == '。' or tag == "!" or tag == "；":
                                output_data.write("\r\n")

            else:
                # print("is D:" + str(word_list))
                isO = True
                pword = re.split(u'[\n]', word_list.replace("<H>", "").replace("</H>", "").replace("<M>", "").replace("</M>", ""))
                for i, pchar in enumerate(pword):

                    if pchar == "<" or pchar == ">" or pchar == "/" or pchar == "D" or pchar == "H":
                        # print(1)
                        del pchar[i]
                        continue
                    if len(pchar) == 0:
                        break
                    if len(pchar) == 1:
                        if pchar == "<" or pchar == ">" or pchar == "/" or pchar == "D" or pchar == "H":
                            del pchar[i]
                        if pchar != "<" and pchar != ">" and pchar != "/" and pchar != "D" and pchar != "H" and pchar != "P":
                            output_data.write(pchar + " B")
                            output_data.write("\r\n")
                            if pchar == '，' or pchar == '。' or pchar == "!" or pchar == "：":
                                output_data.write("\r\n")
                    else:
                        if pchar != "<" and pchar != ">" and pchar != "/" and pchar != "D" and pchar != "H" and pchar != "P":
                            output_data.write(pchar[0] + " B")
                            output_data.write("\r\n")
                            for w in pchar[1:len(pchar)-1]:
                                output_data.write(w + " O")
                                output_data.write("\r\n")
                                if w == '，' or w == '。' or w == "!" or w == "：":
                                    output_data.write("\r\n")
                            output_data.write(pchar[len(pchar)-1] + " E")
                            output_data.write("\r\n")


    output_data.close()

def do_tag1(name, dic, path_B, path_E):
    #提取证据关键字，
    output_data = codecs.open(path_B, 'w', 'utf-8')
    output_data_E = codecs.open(path_E, 'w', 'utf-8')
    for n in name:
        # print(path)
        pi_str = dic[str(n)]
        sentences_B = re.split(u'[<][D][>]', pi_str)
        sentences_E = re.split(u'[<][/][D][>]', pi_str)
        print(sentences_B)
        for i, sentence in enumerate(sentences_B):
            seg = sentence.replace("</D>", "").replace("<H>", "").replace("</H>", "").replace("<M>", "").replace("</M>", "")
            if i != 0:
                output_data.write(seg[0] + " B")
                output_data.write("\r\n")
                for w in seg[1:len(seg)]:
                    output_data.write(w + " O")
                    output_data.write("\r\n")
                    if w == '，' or w == '。' or w == "!" or w == "：":
                        output_data.write("\r\n")
            else:
                for w in seg[0:len(seg)]:
                    output_data.write(w + " O")
                    output_data.write("\r\n")
                    if w == '，' or w == '。' or w == "!" or w == "：":
                        output_data.write("\r\n")

        for i, sentence in enumerate(sentences_E):
            seg = sentence.replace("<D>", "").replace("<H>", "").replace("</H>", "").replace("<M>", "").replace("</M>", "")
            if i == len(sentences_E)-1:
                for w in seg[0:len(seg)]:
                    output_data_E.write(w + " O")
                    output_data_E.write("\r\n")
                    if w == '，' or w == '。' or w == "!" or w == "：":
                        output_data_E.write("\r\n")
            else:
                for w in seg[0:len(seg)-1]:
                    output_data_E.write(w + " O")
                    output_data_E.write("\r\n")
                    if w == '，' or w == '。' or w == "!" or w == "：":
                        output_data_E.write("\r\n")
                output_data_E.write(seg[len(seg) - 1] + " E")
                output_data_E.write("\r\n")


    output_data_E.close()
    output_data.close()

def do_tag_bio(name, dic, path):
    output_data = codecs.open(path, 'w', 'utf-8')

    for n in name:
        # print(n)
        s = []
        d = []
        h = []
        pi_str = dic[str(n)]
        # print("123："+str(p))
        D = re.findall(r"<D>(.+?)</D>", pi_str)  # D表示证据
        # D = re.findall(r"<H>(.+?)</H>", pi_str)  # D表示证据

        sentences = re.split(u'[<][D][>]', pi_str)
        for seg in sentences:
            ss = re.split(u'[<][/][D][>]', seg)
            for i in ss:
                s.append(i)
        print(path, s)
        isO = True
        for word_list in s:  # 所有字符合集
            for D_list in D:
                if word_list == D_list:
                    isO = False
                    break
            if isO:
                # print("isO:"+str(word_list))
                word = re.split(u'[\n]', word_list)
                for char in word:
                    char_list = list(char)
                    for tag in char_list:
                        if tag == "<" or tag == ">" or tag == "/" or tag == "D" or tag == "H" or tag == "P":
                            del tag
                            continue
                        if tag != "<" and tag != ">" and tag != "/" and tag != "D" and tag != "H" and tag != "P" and tag != "M":
                            output_data.write(tag + " O")
                            output_data.write("\r\n")
                            if tag == '，' or tag == '。' or tag == "!" or tag == "：" or tag == "!":
                                output_data.write("\r\n")

            else:
                print("is D:" + str(word_list))
                isO = True
                # p_list = word_list
                pword = re.split(u'[\n]',
                                 word_list.replace("<H>", "").replace("</H>", "").replace("<M>", "").replace("</M>",
                                                                                                             ""))
                for i, pchar in enumerate(pword):

                    if pchar == "<" or pchar == ">" or pchar == "/" or pchar == "D" or pchar == "H":
                        # print(1)
                        del pchar[i]
                        continue
                    if len(pchar) == 0:
                        break
                    if len(pchar) == 1:
                        if pchar == "<" or pchar == ">" or pchar == "/" or pchar == "D" or pchar == "H":
                            del pchar[i]
                        if pchar != "<" and pchar != ">" and pchar != "/" and pchar != "D" and pchar != "H" and pchar != "P":
                            output_data.write(pchar + " B-D")
                            output_data.write("\r\n")
                            if pchar == '，' or pchar == '。' or pchar == "!" or pchar == "：" or pchar == "!":
                                output_data.write("\r\n")
                    else:
                        if pchar != "<" and pchar != ">" and pchar != "/" and pchar != "D" and pchar != "H" and pchar != "P":
                            output_data.write(pchar[0] + " B-D")
                            output_data.write("\r\n")
                            for w in pchar[1:len(pchar)]:
                                if w == "<" or w == ">" or w == "/" or w == "D" or w == "H":
                                    del w
                                    continue
                                output_data.write(w + " I-D")
                                output_data.write("\r\n")
                                if w == '，' or w == '。' or w == "!" or w == "：" or pchar == "!":
                                    output_data.write("\r\n")


    output_data.close()

def do_tag_train(name, dic, path):
    # 提取证据关键字，
    output_data = codecs.open(path, 'w', 'utf-8')

    for n in name:

        pi_str = dic[str(n)]#证据段
        temp_str = pi_str.replace("，", "，|").replace("、", "、|").replace("。", "。|").replace("？", "？|").replace("；","；|").replace("！", "！|").replace("：", "：|").replace(" ", "|").replace(",", ",|")
        sentences = temp_str.split("|")

        for segs in sentences:
            entities = []
            end_ids = []
            entity_temp = re.findall(r"<D>(.+?)</D>", segs)
            # entity_temp = re.findall(r"<H>(.+?)</H>", segs)
            for temp in entity_temp:
                entities.append(temp.replace("</H>", "").replace("<H>", "").replace("</M>", "").replace("<M>", ""))
            seg = segs.replace("</H>", "").replace("<H>", "").replace("</D>", "*").replace("<D>", "").replace("</M>", "").replace("<M>", "")
            seg_lists = list(seg)
            count = 0
            for iid, seg_list in enumerate(seg_lists):
                if seg_list == "*":
                    end_ids.append(iid - count)
                    count += 1
            content = seg.replace("*", "")
            if len(end_ids) != 0:
               for end_id in end_ids:
                   if end_id - 13 < 0:
                       con_block = content[0:end_id]
                   else:
                       con_block = content[len(seg) - 10:end_id-1]
                   for j in range(0, len(con_block)):
                       con_json = {}
                       isEntity = False
                       for entity in entities:
                           if con_block[j:len(con_block)] == entity:
                               isEntity = True
                               break
                       if isEntity:
                           if j - 3 <= 0:
                               con_json["left"] = con_block[0:j]
                           else:
                               con_json["left"] = con_block[j - 3:j]
                           con_json["entity"] = con_block[j:len(con_block)]
                           con_json["right"] = content[end_id]
                           con_json["label"] = "yes"
                       else:
                           if j - 3 <= 0:
                               con_json["left"] = con_block[0:j]
                           else:
                               con_json["left"] = con_block[j - 3:j]
                           con_json["entity"] = con_block[j:len(con_block)]
                           con_json["right"] = content[end_id]
                           con_json["label"] = "no"
                       print(con_json)
                       output_data.write(json.dumps(con_json, ensure_ascii=False) + '\n')

    output_data.close()

def do_tag_train1(name, path):
    # 提取证据关键字，
    output_data = codecs.open(path, 'w', 'utf-8')

    for n in codecs.open(name, "r", "utf-8").readlines():

        pi_str = str(n).strip() #证据段
        temp_str = pi_str.replace("，", "，|").replace("。", "。|").replace("？", "？|").replace("；", "；|").replace("！", "！|").replace("：", "：|").replace(" ", "|").replace(",", ",|")
        sentences = temp_str.split("|")

        for segs in sentences:
            entities = []
            end_ids = []
            entity_temp = re.findall(r"<D>(.+?)</D>", segs)
            # entity_temp = re.findall(r"<H>(.+?)</H>", segs)
            for temp in entity_temp:
                entities.append(temp.replace("</H>", "").replace("<H>", "").replace("</M>", "").replace("<M>", ""))
            seg = segs.replace("</H>", "").replace("<H>", "").replace("</D>", "*").replace("<D>", "").replace("</M>", "").replace("<M>", "")

            seg_lists = list(seg)
            count = 0
            for iid, seg_list in enumerate(seg_lists):
                if seg_list == "*":
                    count += 1
                    end_ids.append(iid - count)

            content = seg.replace("*", "")
            if len(end_ids) != 0:
                for end_id in end_ids:
                    # count = 0
                    if end_id - 13 < 0:
                        con_block = content[0:end_id+1]
                    else:
                        con_block = content[end_id - 13:end_id + 1]
                    for j in range(0, len(con_block)):
                        con_json = {}
                        isEntity = False
                        for entity in entities:
                            if con_block[j:len(con_block)] == entity:
                                isEntity = True
                                break
                        if isEntity:
                            if j - 3 <= 0:
                                con_json["left"] = con_block[0:j]
                            else:
                                con_json["left"] = con_block[j - 3:j]
                            con_json["entity"] = con_block[j:len(con_block)]
                            con_json["right"] = content[end_id+1]
                            con_json["label"] = "yes"
                        else:
                            if j - 3 <= 0:
                                con_json["left"] = con_block[0:j]
                            else:
                                con_json["left"] = con_block[j - 3:j]
                            con_json["entity"] = con_block[j:len(con_block)]
                            con_json["right"] = content[end_id+1]
                            con_json["label"] = "no"
                        print(con_json)
                        output_data.write(json.dumps(con_json, ensure_ascii=False) + '\n')
                    # if count < 10:
                    #     for i in range(10 - count):
                    #         con_json = {}
                    #         con_json["left"] = "*"
                    #         con_json["entity"] = "*"
                    #         con_json["right"] = "*"
                    #         con_json["label"] = "no"
                    #         # print(con_json)
                    #         output_data.write(json.dumps(con_json, ensure_ascii=False) + '\n')
    output_data.close()

def do_tag_test(name, dic, path, model, sess, word_to_id, tag_to_id):
    output_data = codecs.open(path, 'a', 'utf-8')
    for n in name:
        pi_str = dic[str(n)]  # 证据段
        temp_str = pi_str.replace("，", "，|").replace("。", "。|").replace("、", "、|").replace("？", "？|").replace("；", "；|").replace("！", "！|").replace("：", "：|").replace(" ", "|").replace(",", ",|")
        contents = temp_str.split("|")
        for segs in contents:
            entities = []
            entity_temp = re.findall(r"<D>(.+?)</D>", segs)
            # entity_temp = re.findall(r"<H>(.+?)</H>", segs)
            for temp in entity_temp:
                entities.append(temp.replace("</H>", "").replace("<H>", "").replace("</M>", "").replace("<M>", ""))
            seg = segs.replace("</H>", "").replace("<H>", "").replace("</D>", "").replace("<D>", "").replace("</M>", "").replace("<M>", "")

            if seg != "":
                tags = model.predict(sess, input_from_line(seg, word_to_id), tag_to_id)
                end_tags = []
                begin_tags = []
                for i, tag in enumerate(tags):
                    if tag == 2:
                        end_tags.append(i)
                    if tag == 1:
                        begin_tags.append(i)

                if len(end_tags) != 0 and len(begin_tags) != 0:
                    try:
                        for end_tag in end_tags:
                            # if end_tag-13 >= 0:
                            #     begin = end_tag-10
                            # else:
                            #     begin = 0
                            # for b in range(begin, end_tag):
                            #     con_block = seg[b:end_tag + 1]
                            for begin_tag in begin_tags:
                                if end_tag > begin_tag:
                                    con_block = seg[begin_tag:end_tag + 1]
                                else:
                                    continue
                                con_json = {}
                                isEntity = False
                                for entity in entities:
                                    if con_block == entity:
                                        isEntity = True
                                        break
                                if isEntity:
                                    # con_json["left"] = seg[int(begin_tag) - 3:int(begin_tag)]
                                    if begin_tag - 3 <= 0:
                                        con_json["left"] = seg[0:int(begin_tag)]
                                    else:
                                        con_json["left"] = seg[int(begin_tag) - 3:int(begin_tag)]
                                    con_json["entity"] = con_block
                                    con_json["right"] = seg[int(end_tag)+1]
                                    con_json["label"] = "yes"
                                else:
                                    # con_json["left"] = seg[int(begin_tag) - 3:int(begin_tag)]
                                    if begin_tag - 3 <= 0:
                                        con_json["left"] = seg[0:int(begin_tag)]
                                    else:
                                        con_json["left"] = seg[int(begin_tag) - 3:int(begin_tag)]
                                    con_json["entity"] = con_block
                                    con_json["right"] = seg[int(end_tag) + 1]
                                    con_json["label"] = "no"
                                output_data.write(json.dumps(con_json, ensure_ascii=False) + '\n')
                                # print(con_json)
                    except IndexError:

                        print(entity)
                        pass
    output_data.close()

def do_tag_test_all(name, path, model, sess, word_to_id, tag_to_id):
    output_data = codecs.open(path, 'w', 'utf-8')
    for n in codecs.open(name, "r", "utf-8").readlines():
        pi_str = str(n).strip()  # 证据段
        temp_str = pi_str.replace("，", "，|").replace("：", "：|").replace("。", "。|").replace("、", "、|").replace("？", "？|").replace("；", "；|").replace("！", "！|").replace("：", "：|").replace(" ", "|").replace(",", ",|")
        contents = temp_str.split("|")
        for segs in contents:
            entities = []
            entity_temp = re.findall(r"<D>(.+?)</D>", segs)
            # entity_temp = re.findall(r"<H>(.+?)</H>", segs)
            for temp in entity_temp:
                entities.append(temp.replace("</H>", "").replace("<H>", "").replace("</M>", "").replace("<M>", ""))
            seg = segs.replace("</H>", "").replace("<H>", "").replace("</D>", "").replace("<D>", "").replace("</M>", "").replace("<M>", "")

            if seg != "":
                tags = model.predict(sess, input_from_line(seg, word_to_id), tag_to_id)
                end_tags = []
                begin_tags = []
                for i, tag in enumerate(tags):
                    if tag == 2:
                        end_tags.append(i)
                    if tag == 1:
                        begin_tags.append(i)

                if len(end_tags) != 0 and len(begin_tags) != 0:
                    try:
                        for end_tag in end_tags:

                            # if end_tag-10 >= 0:
                            #     begin = end_tag-10
                            # else:
                            #     begin = 0
                            # for b in range(begin, end_tag):
                            #
                            #     con_block = seg[b:end_tag + 1]
                            for begin_tag in begin_tags:
                                if end_tag > begin_tag:
                                    con_block = seg[begin_tag:end_tag + 1]
                                else:
                                    continue
                                con_json = {}
                                isEntity = False
                                for entity in entities:
                                    if con_block == entity:
                                        isEntity = True
                                        break
                                if isEntity:
                                    if begin_tag == 0:
                                        con_json["left"] = "、"
                                    elif begin_tag - 3 <= 0:
                                        con_json["left"] = seg[0:int(begin_tag)]
                                    else:
                                        con_json["left"] = seg[int(begin_tag) - 3:int(begin_tag)]
                                    # con_json["left"] = seg[int(b) - 3:int(b)]
                                    con_json["entity"] = con_block
                                    con_json["right"] = seg[int(end_tag)+1]
                                    con_json["label"] = "yes"
                                else:
                                    # con_json["left"] = seg[int(b) - 3:int(b)]
                                    if begin_tag - 3 <= 0:
                                        con_json["left"] = seg[0:int(begin_tag)]
                                    else:
                                        con_json["left"] = seg[int(begin_tag) - 3:int(begin_tag)]
                                    con_json["entity"] = con_block
                                    con_json["right"] = seg[int(end_tag) + 1]
                                    con_json["label"] = "no"
                                output_data.write(json.dumps(con_json, ensure_ascii=False) + '\n')
                    except IndexError:
                        print(entity)
                        pass
    output_data.close()

if __name__ == "__main__":
    input_data = 'all/'
    dic, name = character_tagging(input_data)

    train, dev, test = do_random(name)
    output_data1 = codecs.open("all/train.utf8", 'w', 'utf-8')
    output_data2 = codecs.open("all/dev.utf8", 'w', 'utf-8')
    output_data3 = codecs.open("all/test.utf8", 'w', 'utf-8')
    for train_ in train:
        # print(dic[train_])
        output_data1.write(dic[train_] + "\n" + "\n")
    output_data1.close()

    for dev_ in dev:
        output_data2.write(dic[dev_] + "\n" + "\n")
    output_data2.close()

    for test_ in test:
        output_data3.write(dic[test_] + "\n" + "\n")
    output_data3.close()

    do_tag(train, dic, "data/boundaryIdentify/evidence/train-h.utf8")
    do_tag(dev, dic, "data/boundaryIdentify/evidence/dev-h.utf8")
    do_tag(test, dic, "data/boundaryIdentify/evidence/test-h.utf8")

    # do_tag1(train, dic, "data/boundaryIdentify/evidence/train-B.utf8", "data/boundaryIdentify/evidence/train-E.utf8")
    # do_tag1(dev, dic, "data/boundaryIdentify/evidence/dev-B.utf8", "data/boundaryIdentify/evidence/dev-E.utf8")
    # do_tag1(test, dic, "data/boundaryIdentify/evidence/test-B.utf8", "data/boundaryIdentify/evidence/test-E.utf8")

    # # bi-lstm-crf NER 数据集
    do_tag_bio(train, dic, "data/boundaryIdentify/lstm_data/train-h.utf8")
    do_tag_bio(dev, dic, "data/boundaryIdentify/lstm_data/dev-h.utf8")
    do_tag_bio(test, dic, "data/boundaryIdentify/lstm_data/test-h.utf8")

    # input_data = 'all/'
    # dic, name = character_tagging(input_data)
    # train, dev, test = do_random(name)
    # output_data1 = codecs.open("all/train.utf8", 'w', 'utf-8')
    # output_data2 = codecs.open("all/dev.utf8", 'w', 'utf-8')
    # output_data3 = codecs.open("all/test.utf8", 'w', 'utf-8')
    # for train_ in train:
    #     output_data1.write(dic[train_] + "\n" + "\n")
    # output_data1.close()
    #
    # for dev_ in dev:
    #     output_data2.write(dic[dev_] + "\n" + "\n")
    # output_data2.close()
    #
    # for test_ in test:
    #     output_data3.write(dic[test_] + "\n" + "\n")
    # output_data3.close()
    #
    #
    # do_tag(train, dic, "data/boundaryIdentify/evidence/train-h.utf8")
    # do_tag(dev, dic, "data/boundaryIdentify/evidence/dev-h.utf8")
    # do_tag(test, dic, "data/boundaryIdentify/evidence/test-h.utf8")
    # # # bi-lstm-crf NER 数据集
    # do_tag_bio(train, dic, "data/boundaryIdentify/lstm_data/train-h.utf8")
    # do_tag_bio(dev, dic, "data/boundaryIdentify/lstm_data/dev-h.utf8")
    # do_tag_bio(test, dic, "data/boundaryIdentify/lstm_data/test-h.utf8")
    #
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 设置GPU
    # boundary_config = Config()
    # boundary_word_to_id, boundary_tag_to_id = build_vocabs(boundary_config)
    # # 修正字典大小
    # boundary_config.vocab_size = len(boundary_word_to_id)
    # boundary_model = BiDirectionalRNNModel(boundary_config)
    # # 训练边界识别模型
    # if boundary_config.train_model:
    #     boundary_train(boundary_config, boundary_word_to_id, boundary_tag_to_id, boundary_model)
    # # 利用边界识别模型构造多核CNN的数据分类数据集
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # with tf.Session(config=tf_config) as sess:
    #     boundary_model = create_model(sess, boundary_model, load_word2vec, boundary_config, boundary_word_to_id)
    #     # do_tag_train(train, dic, "data/entityClassifier/evidence/BE/train.json")
    #     # do_tag_test(dev, dic, "data/entityClassifier/evidence/BE/val.json", boundary_model, sess, boundary_word_to_id,
    #     #             boundary_tag_to_id)
    #     # do_tag_test(test, dic, "data/entityClassifier/evidence/BE/test.json", boundary_model, sess, boundary_word_to_id,
    #     #             boundary_tag_to_id)
    #     do_tag_train1("all/train.utf8", "data/entityClassifier/evidence/BE/train.json")
    #
    #     do_tag_test_all("all/dev.utf8", "data/entityClassifier/evidence/BE/val.json", boundary_model, sess, boundary_word_to_id,
    #                 boundary_tag_to_id)
    #     do_tag_test_all("all/test.utf8", "data/entityClassifier/evidence/BE/test.json", boundary_model, sess, boundary_word_to_id,
    #                 boundary_tag_to_id)
    #
    #
    # # 多核CNN的证据分类模型
    # classify_config = TCNNConfig()
    # classify_word_to_id, classify_tag_to_id = classifier_build_vocabs(classify_config)
    # # 修正分类器的词典大小
    # classify_config.batch_size = len(classify_word_to_id)
    # classify_model = multi_cnn_classifier(classify_config)
    # # 训练多核CNN分类模型
    # if classify_config.train_model:
    #     classifier_train(classify_config, classify_word_to_id, classify_tag_to_id, classify_model, boundary_config)
