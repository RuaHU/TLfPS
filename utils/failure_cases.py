#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 20:12:27 2020

@author: hu
"""
import pickle, os
import numpy as np
import cv2
import sys, getopt
parpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
curpath = os.path.dirname(os.path.abspath(__file__))

def EPRW(tape, PRW):
    failure_cases = []
    for key in tape.keys():
        pimg = os.path.join(PRW, 'frames', key)
        qimg = cv2.imread(pimg)
        qid, qbox, ap, acc, recall_rate, score, true, gboxes, gimname = tape[key]
        x, y, w, h = np.array(qbox).astype(np.int32)
        re_qimg = cv2.resize(qimg[y:y+h, x:x+w], (64, 128), interpolation=cv2.INTER_LINEAR)
        case = [qid, re_qimg]
        indices = np.where(tape[key][6]==1)[0]
        match_case = []
        if indices.size > 0:
            for index in indices:
                imname = tape[key][8][index]
                pimg = os.path.join(PRW, 'frames', imname)
                image = cv2.imread(pimg)
                x1, y1, x2, y2 = np.array(tape[key][7][index]).astype(np.int32)
                s = tape[key][5][index]
                r_img = cv2.resize(image[y1:y2, x1:x2], (64, 128), interpolation=cv2.INTER_LINEAR)
                match_case.append([index, s, r_img])
                break
        
        case.append(match_case)
        imname = tape[key][8][0]
        pimg = os.path.join(PRW, 'frames', imname)
        image = cv2.imread(pimg)
        x1, y1, x2, y2 = np.array(tape[key][7][0]).astype(np.int32)
        s = tape[key][5][0]
        r_img = cv2.resize(image[y1:y2, x1:x2], (64, 128), interpolation=cv2.INTER_LINEAR)
        case.append([0, s, r_img])
        failure_cases.append(case)
    
    
    rows, cols = 8, 6
    assert rows * cols <= 50
    
    height = 128 * rows + 32 * (rows - 1)
    width = 64 * 3 * cols + 2 * cols * 6 + 16 * (cols - 1)
    canvas = np.ones([height, width, 3], dtype = np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            y = i * (128 + 32)
            x = j * (64 * 3 + 2 * 6 + 16)
            qid, re_qimg, match_case, false_case = failure_cases[i*cols + j]
            canvas[y:y+128, x:x+64] = re_qimg
            y1, x1 = y, x + 64 + 6
            canvas[y1:y1+128, x1:x1+64] = match_case[0][2]
            y2, x2 = y1, x1 + 64 + 6
            canvas[y2:y2 + 128, x2:x2 + 64] = false_case[2]
            print(i, j, qid, match_case[0][0], match_case[0][1], false_case[1])
    
    cv2.imwrite('prw_canvas.jpg', canvas)


def ECUHK_SYSU(tape, CUHK):
    failure_cases = []
    for key in tape.keys():
        pimg = os.path.join(CUHK, 'dataset/Image/SSM', key)
        qimg = cv2.imread(pimg)
        qid, qbox, ap, acc, recall_rate, score, true, gboxes, gimname = tape[key]
        x, y, w, h = np.array(qbox[0]).astype(np.int32)
        re_qimg = cv2.resize(qimg[y:y+h, x:x+w], (64, 128), interpolation=cv2.INTER_LINEAR)
        case = [qid, re_qimg]
        indices = np.where(tape[key][6]==1)[0]
        match_case = []
        if indices.size > 0:
            for index in indices:
                imname = tape[key][8][index]
                pimg = os.path.join(CUHK, 'dataset/Image/SSM', imname)
                image = cv2.imread(pimg)
                x1, y1, x2, y2 = np.array(tape[key][7][index]).astype(np.int32)
                s = tape[key][5][index]
                r_img = cv2.resize(image[y1:y2, x1:x2], (64, 128), interpolation=cv2.INTER_LINEAR)
                match_case.append([index, s, r_img])
                break
        case.append(match_case)
        imname = tape[key][8][0]
        pimg = os.path.join(CUHK, 'dataset/Image/SSM', imname)
        image = cv2.imread(pimg)
        
        x1, y1, x2, y2 = np.where(tape[key][7][0] < 0, 0, tape[key][7][0]).astype(np.int32)
        s = tape[key][5][0]
        r_img = cv2.resize(image[y1:y2, x1:x2], (64, 128), interpolation=cv2.INTER_LINEAR)
        case.append([0, s, r_img])
        failure_cases.append(case)
        
    rows, cols = 8, 6
    assert rows * cols <= 50
    
    height = 128 * rows + 32 * (rows - 1)
    width = 64 * 3 * cols + 2 * cols * 6 + 16 * (cols - 1)
    canvas = np.ones([height, width, 3], dtype = np.uint8) * 255
    for i in range(rows):
        for j in range(cols):
            y = i * (128 + 32)
            x = j * (64 * 3 + 2 * 6 + 16)
            qid, re_qimg, match_case, false_case = failure_cases[i*cols + j]
            
            canvas[y:y+128, x:x+64] = re_qimg
            y1, x1 = y, x + 64 + 6
            if len(match_case) > 0:
                print(i, j, qid, match_case[0][0], match_case[0][1], false_case[1])
                canvas[y1:y1+128, x1:x1+64] = match_case[0][2]
            else:
                print(i, j, qid, -1, -1, false_case[1])
                canvas[y1:y1+128, x1:x1+64] = np.ones([128, 64, 3], dtype = np.uint8) * 192
            y2, x2 = y1, x1 + 64 + 6
            canvas[y2:y2 + 128, x2:x2 + 64] = false_case[2]
    
    cv2.imwrite('cuhk_canvas.jpg', canvas)

def main(argv):
    M = 'yolov3'
    gpu = '0'
    CUHK_SYSU = "/home/ronghua/Projects/data/dataset-v2/"
    PRW = '/home/ronghua/Projects/data/PRW-v16.04.20/'
    experiment_name = 'default'
    try:
        opts, args = getopt.getopt(argv[1:], 'hm:g:p:c:e:', ['m=', 'gpu=', 'prw=', 'cuhk=', 'en='])
    except getopt.GetoptError:
        print(argv[0] + ' -m <M> -g <gpu>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(argv[0] + ' -m <M> -g <gpu>')
        elif opt in ['-m', '--M']:
            M = arg
        elif opt in ['-p', '--path']:
            PRW = arg
        elif opt in ['-c', '--cuhk']:
            CUHK_SYSU = arg
        elif opt in ['-g', '--gpu']:
            gpu = arg
        elif opt in ['-e', '--en']:
            experiment_name = arg
    
    print('model: [%s] gpu: [%s], CUHK_SYSU: [%s], PRW: [%s]'%(M, gpu, CUHK_SYSU, PRW))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    
    dataset = False
    if os.path.exists(CUHK_SYSU):
        print('processing CUHK-SYSU...')
        filepath = os.path.join(parpath, 'experiment_results', experiment_name, 'cuhk_%s_tape.pkl'%M)
        f = open(filepath, 'rb')
        tape = pickle.load(f)
        ECUHK_SYSU(tape, CUHK_SYSU)
    if os.path.exists(PRW):
        print('processing PRW...')
        filepath = os.path.join(parpath, 'experiment_results', experiment_name, 'prw_%s_tape.pkl'%M)
        f = open(filepath, 'rb')
        tape = pickle.load(f)
        EPRW(tape, PRW)
    
    
if __name__ == "__main__":
   main(sys.argv)
