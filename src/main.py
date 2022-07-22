from math import log
import operator
import cv2
import numpy as np
from helper import make_data_set, construct_decision_tree


if __name__ == '__main__':
    # make_data_set.cons_data_set()
    construct_decision_tree.making_tree()
    '''img = cv2.imread('/Users/cuiguangyuan/Desktop/train_img/img16.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    threshed = make_data_set.threshing(img)
    after = make_data_set.resize_img(img.shape[0], img.shape[1], threshed)
    final_img = make_data_set.threshing(after)
    cv2.imshow('tmp window1', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''