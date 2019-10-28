import codecs
import os

def performance_on_positives(true_cls, pred_cls, path):

    # 读取正确的标签
    yes_id = 0
    for label in codecs.open("entityClassifier_vocabs/label.txt", 'r', 'utf-8').readlines():
        label = str(label).strip()
        if label == "yes":
            break
        else:
            yes_id = 1
            break
    print("yes_id:", yes_id)
    if len(true_cls) != len(pred_cls):
        print("y_pred_index and y_test_index should be equal...")
        os.system("pause")

    true_positives = 0 # 正确的实体
    pred_positives = 0 # 预测正确的实体
    correct_num = 0
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip().split()
        if len(line) == 2:
            if line[-1] == "B":
                true_positives += 1
    # 统计预测为正确的实体
    for pred in pred_cls:
        if pred == yes_id:
            pred_positives += 1

    for key, value in enumerate(pred_cls):
        if value == yes_id and true_cls[key] == pred_cls[key]:
            correct_num += 1
    print("-------------------------Performance---------------------------------")
    print("True Positives: ", str(true_positives))
    print("Found Positives:" + str(pred_positives))
    print("Correct Positives:" + str(correct_num))
    P = 0 if pred_positives == 0 else 100. * correct_num / pred_positives
    R = 0 if true_positives == 0 else 100. * correct_num / true_positives
    F = 0 if P + R == 0 else 2 * P * R / (P + R)
    print("Precision: %.2f" % (P))
    print("Recall: %.2f" % (R))
    print("F1: %.2f" % (F))
    return P, R, F
