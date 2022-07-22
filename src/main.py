import numpy as np
from helper import make_data_set, construct_decision_tree
import cv2


def get_feat_labels():
    lst = []
    for i in range(0, 64):
        lst.append('Row_'+str(i))
    return lst


def img2data(img):
    vec = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                vec.append('1')
            else:
                vec.append('0')
    return vec


def main():
    # make_data_set.cons_data_set()
    # construct_decision_tree.making_tree()
    tree_model = construct_decision_tree.grab_tree('../model/tree.txt')
    img = cv2.imread("../data_set/train_img/img33.png", cv2.IMREAD_GRAYSCALE)
    after = make_data_set.process_img(img)
    res = construct_decision_tree.classify(tree_model, get_feat_labels(), img2data(after))
    print('识别结果为: ' + res)


if __name__ == '__main__':
    main()
