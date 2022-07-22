import os
import numpy as np
import pandas as pd
import cv2


def process_img(img):
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    threshed = threshing(img)
    after = resize_img(img.shape[0], img.shape[1], threshed)
    final_img = threshing(after)
    return final_img


def resize_img(height, width, img):
    left = width
    right = 0
    upper = height
    lower = 0
    for i in range(height):
        for j in range(width):
            if img[i][j] == 0:
                left = min(left, j)
                right = max(right, j)
                upper = min(upper, i)
                lower = max(lower, i)
    inner = img[upper:lower, left:right]
    return cv2.resize(inner, (8, 8), interpolation=cv2.INTER_AREA)


def sub_data(img):
    ret = []
    for i in range(8):
        sub = "".join(img[i])
        ret.append(int(sub, 2))
    return ret


def threshing(img):
    ret = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 122:
                ret[i][j] = 255
            else:
                ret[i][j] = 0
    return ret


def get_image_data(img):
    rows = []
    for i in range(0, 8):
        lst = []
        for j in range(0, 8):
            if img[i][j] == 255:
                lst.append('1')
            else:
                lst.append('0')
        rows.append(",".join(lst))
        #rows.append(int("".join(lst), 2))
    return rows


def cons_data_set():
    csv_path = '../data_set/train.csv'
    img_path = '../data_set/train_img/'
    images = os.listdir(img_path)
    img_data = pd.read_csv(csv_path)
    img_data.fillna(0)
    for image in images:
        if image[0:3] == 'img':
            if image[4] == '.':
                num = int(image[3])
            else:
                num = int(image[3:5])
            tmp = process_img(cv2.imread(img_path+image, cv2.IMREAD_GRAYSCALE))
            dt = get_image_data(tmp)
            for index in range(0, 8):
                idx = 'Row_'+str(index)
                img_data.loc[num, idx] = dt[index]

    img_data.to_csv(csv_path)

    print('Constructing new train data set success!')
