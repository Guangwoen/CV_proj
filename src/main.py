import construct_decision_tree, make_data_set
import cv2
import tkinter as tk
from tkinter import filedialog


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
    root = tk.Tk()
    root.withdraw()
    f_path = filedialog.askopenfilename()
    img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
    after = make_data_set.process_img(img)
    tree_model = construct_decision_tree.grab_tree('../model/tree.txt')
    res = construct_decision_tree.classify(tree_model, get_feat_labels(), img2data(after))
    print('识别结果为: ' + res)


if __name__ == '__main__':
    main()
