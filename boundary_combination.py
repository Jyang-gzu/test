import codecs
def combination_model(boundary_tags):
    # entities, combination_output = list(), list()
    # for boundary_tag in boundary_tags:
    #     sentence = boundary_tag[0]
    #     tags = boundary_tag[1]
    #     end_tags = []
    #     begin_tags = []
    #     for i, tag in enumerate(tags):
    #         if tag == 2:
    #             end_tags.append(i)
    #         if tag == 1:
    #             begin_tags.append(i)
    #
    #     if len(end_tags) != 0 and len(begin_tags) != 0:
    #         for end_tag in end_tags:
    #             for begin_tag in begin_tags:
    #                 if end_tag > begin_tag:
    #                     entity = sentence[begin_tag:end_tag + 1]
    #                 else:
    #                     continue
    #                 entities.append(entity)
    #                 con_json = {}
    #                 if begin_tag - 3 < 3:
    #                     con_json["left"] = sentence[0:int(begin_tag)]
    #                 else:
    #                     con_json["left"] = sentence[int(begin_tag) - 3:int(begin_tag)]
    #                 con_json["entity"] = entity
    #                 con_json["right"] = sentence[int(end_tag) + 1]
    #                 combination_output.append(con_json)
    # return entities, combination_output
    output_data1 = codecs.open("data/boundaryIdentify/fenci/result.utf8", 'a', 'utf-8')
    for boundary_tag in boundary_tags:
        sentence = boundary_tag[0]
        tags = boundary_tag[1]
        for i, tag in enumerate(tags):
            if tag == 0:
                output_data1.write(sentence[i])
            elif tag == 3:
                output_data1.write(sentence[i])
            elif tag == 1:
                output_data1.write(sentence[i] + '  ')
            else:  # tag == 'S'
                output_data1.write(sentence[i] + '  ')
    output_data1.write("\r\n")

