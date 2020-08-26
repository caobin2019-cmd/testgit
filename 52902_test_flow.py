#!/usr/bin/env python
# encoding:utf-8
"""
author: caobin
@contact: caobin@unionbigdata.com
@software:
@file: deploy_result_final.py
@time: 2020/07/24
@desc:
"""
import os
from glob import glob
from mmdet.apis import init_detector, inference_detector
from sklearn.metrics import confusion_matrix
# from metrics.test_result_analysis import ResultAnalysis
from metrics.dt_gt_analysis import ResultAnalysis
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import cv2
import pickle
import pandas as pd
from lxml.etree import Element, SubElement, tostring, ElementTree
import time



def initModel(deploy_path, out_dir_path, device='cuda:0'):
    global model
    global labels
    global config

    config_path = os.path.join(out_dir_path, '20200824_133033.py')
    model_path = os.path.join(out_dir_path, 'epoch_24.pth')
    class_path = os.path.join(deploy_path, 'classes.txt')
    json_path = os.path.join(deploy_path, 'rule_52902_confidence.json')

    with open(json_path, 'r') as f:
        config = json.load(f)
    labels = []
    for line in open(class_path, 'r'):
        lineTemp = line.strip()
        if lineTemp:
            labels.append(lineTemp)

    model = init_detector(config_path, model_path, device=device)


def model_test(result,
               img_name,
               codes,
               score_thr=0.05):

    output_bboxes = []
    json_dict = []

    total_bbox = []
    for id, boxes in enumerate(result):  # loop for categories
        category_id = id + 1
        if len(boxes) != 0:
            for box in boxes:  # loop for bbox
                conf = box[4]
                if conf > score_thr:
                    total_bbox.append(list(box) + [category_id])

    bboxes = np.array(total_bbox)
    best_bboxes = bboxes
    output_bboxes.append(best_bboxes)
    for bbox in best_bboxes:
        coord = [round(i, 2) for i in bbox[:4]]
        area = (coord[2]-coord[0])*(coord[3]-coord[1])
        conf, category = bbox[4], codes[int(bbox[5]) - 1]
        json_dict.append({'name': img_name, 'category': category, 'bbox': coord, 'score': conf, 'bbox_score': bbox[:5], 'area': area})

    return json_dict


def draw_pred_bounding_boxes(img, bounding_boxes_of_image, confidence_threshold):
    confidences = []
    height = img.shape[0]
    width = img.shape[1]
    i = 0

    for bounding_box_of_image in bounding_boxes_of_image:
        xmin = bounding_box_of_image[0]
        ymin = bounding_box_of_image[1]
        xmax = bounding_box_of_image[2]
        ymax = bounding_box_of_image[3]
        confidence = bounding_box_of_image[4]
        bbox_category = bounding_box_of_image[5]
        confidences.append(confidence)

        if confidence > confidence_threshold:
            i += 1
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), thickness=3)
            text_cord = (width-200, 30*i)
            cv2.line(img, (xmax, ymin), (text_cord), (0, 255, 0), 1)
            cv2.putText(img, bbox_category + ':{:.3f}'.format(confidence), text_cord,
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 255, 0))


    return img, confidences


def draw_true_bounding_boxes(img, true_bounding_boxes, bbox_categories):
    """
    draw the true bounding boxes on the image
    """
    height = img.shape[0]
    width = img.shape[1]

    for true_bounding_box, bbox_category in zip(true_bounding_boxes, bbox_categories):
        xmin = true_bounding_box[0]
        ymin = true_bounding_box[1]
        xmax = xmin + true_bounding_box[2]
        ymax = ymin + true_bounding_box[3]
        bbox_category_name = bbox_category

        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness=3)

        text_cord = (min(int(xmin), width - 150), max(ymin, 45))
        cv2.putText(img, bbox_category_name, text_cord,
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255))
    return img


def draw_bounding_boxes(true_categories, prediction_categories, predition_bounding_boxes,
                        img_path, save_root, detect_result):
    for true_category, prediction_category, prediction_bounding_box, path, det_df in tqdm(zip(true_categories, prediction_categories,
                                                                                 predition_bounding_boxes, img_path, detect_result)):
        img = cv2.imread(path)
        # if img is None:
        #     continue
        img_name = path.split('/')[-1]
        height, width = img.shape[0], img.shape[1]
        if prediction_category:
            try:
                xmin = prediction_bounding_box[0]
                ymin = prediction_bounding_box[1]
                xmax = prediction_bounding_box[2]
                ymax = prediction_bounding_box[3]
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), thickness=3)
                cv2.line(img, (int(xmax), int(ymin)), (int(width - 200), 50), (0, 0, 255), thickness=1)
                cv2.line(img, (int(xmax), int(ymin)), (int(width - 200), 80), (0, 0, 255), thickness=1)
                cv2.putText(img, 'pred:{}'.format(prediction_category), (int(width-200), 50),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
                cv2.putText(img, 'true:{}'.format(true_category), (int(width - 200), 80),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
            except:
                print(prediction_bounding_box)
        if det_df is not None:
            count = 0
            for row in det_df.iterrows():
                bbox = row[1]['bbox']
                cate = row[1]['category']
                score = row[1]['score']
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
                cv2.rectangle(img, (int(xmin)-2*count, int(ymin)-2*count), (int(xmax) + 2*count, int(ymax) + 2*count), (0, 255, 0), thickness=3)
                cv2.line(img, (int(xmin) - 2*count, int(ymin)- 2*count), (int(0), 50+count*30), (0, 255, 0), thickness=1)
                cv2.putText(img, '{}:{}'.format(cate, round(score, 2)), (int(0), 50+count*30),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
                count += 1
        name_str = 'draw_out_{}'.format(time.strftime('%Y-%m-%d', time.localtime()))
        if not os.path.exists(os.path.join(save_root, name_str, true_category, prediction_category)):
            os.makedirs(os.path.join(save_root, name_str, true_category, prediction_category))
        cv2.imwrite(os.path.join(save_root, name_str, true_category, prediction_category, img_name), img)


def final_labeling(df,code,bbox,score, obj, img_path):

    if df.shape[0]==0:
        print('~~~~~~~~~~~~~~~')
        return True
    name = df['name'][0]
    # im_path = os.path.join(img_path, name)
    img = cv2.imread(img_path)
    height = img.shape[0]
    width = img.shape[1]
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    xmin = (bbox[0]).astype('int')
    ymin = (bbox[1]).astype('int')
    xmax = (bbox[2]).astype('int')
    ymax = (bbox[3]).astype('int')
    category = code
    node_object = SubElement(node_root, 'object')
    node_name = SubElement(node_object, 'name')
    node_name.text = category
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    node_bndbox = SubElement(node_object, 'bndbox')
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(xmin)
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(ymin)
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(xmax)
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(ymax)
    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    # dom = parseString(xml)
    namex = name.split('.')[0] + '.xml'
    xml_name = os.path.join(obj, namex)
    tree = ElementTree(node_root)
    tree.write(xml_name, pretty_print=True, xml_declaration=True, encoding='utf-8')
    return False


def read_json_file(filepath):
    with open(filepath, 'r', encoding='UTF-8') as f:
        json_file = json.load(f)
    return json_file


def cm2df(conf_matrix, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata = {}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label] = conf_matrix[i, j]
        df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
    return df[labels]


def save_confusion_matrix(true_categories, prediction_categories, label, save_file):
    conf_matrix = confusion_matrix(true_categories, prediction_categories, labels=label, sample_weight=None)
    save_path, file_name = os.path.split(save_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cm_df = cm2df(conf_matrix, label)

    precision_list = []
    recall_list = []

    for i in range(len(cm_df.columns)):
        precision = round(cm_df.iloc[i, i] / sum(cm_df[cm_df.columns[i]]), 4)
        recall = round(cm_df.iloc[i, i] / sum(cm_df.loc[cm_df.columns[i]]), 4)
        precision_list.append(precision)
        recall_list.append(recall)

    cm_df['recall'] = recall_list
    precision_list.append(None)
    cm_df.loc['precision'] = precision_list
    cm_df.to_csv(save_file)


def deploy_result_final_csv(sample_root,deploy_path, out_dir_path, out_path=None, result_path=None, device='cuda:0', **kwargs):
    initModel(deploy_path, out_dir_path, device=device)
    df = pd.read_csv(os.path.join('/data/sdv2/cb/work_dir/20200727_52902/', 'test_data_0804.csv'))
    label = ['DRM1', 'DRMO', 'DRMX', 'IRP1', 'ISC3', 'MDF1', 'MPL1', 'MPLO', 'MPT3', 'MPTO', 'MRM3',
             'MSC2', 'TRM1', 'TRMO', 'TRMX', 'WPT1', 'FALSE', 'Others']
    img_lst = []
    size_list = []
    for item in df.iterrows():
        img_lst.append(item[1]['img_path'])
        size_list.append(item[1]['size'])
    pbar = tqdm(img_lst)
    # get size
    log = []
    size = kwargs.get('size', None)
    true_categories = []
    prediction_categories = []
    true_bounding_boxes = []
    detect_result = []
    # iter img
    for i, img_path in enumerate(pbar):
        img_name = img_path.split('/')[-1]
        category = img_path.split('/')[-2]
        size = size_list[i]
        img = cv2.imread(img_path)
        if img is None:
            print('img {} is None'.format(img_path))
            # continue
        result = inference_detector(model, img_path)
        json_dict = model_test(result, img_name, labels)
        det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score', 'area'])
        # prior parameters
        prior_lst = config['prior_order']
        if config['false_name'] not in prior_lst:
            prior_lst.append(config['false_name'])
        if config['other_name'] not in prior_lst:
            prior_lst.append(config['other_name'])
        thr_by_code = config['thr_by_code']
        for code in labels:
            if code not in thr_by_code:
                thr_by_code[code] = config["other_thr"]
        if len(det_df) > 0:
            max_idx = det_df.score.idxmax()
            max_conf = det_df.score[max_idx]
            predict = det_df.loc[max_idx, 'category']
            for j, row in det_df.iterrows():
                predict_idx = prior_lst.index(predict)
                code = row['category']
                code_idx = prior_lst.index(code)
                score = row['score']
                if code_idx < predict_idx and score > max_conf * 0.8:
                    predict = code
            max_conf = max(det_df.loc[det_df.category == predict, 'score'].values)
            select_df = det_df[(det_df['category'] == predict) & (det_df['score'] == max_conf)]
            if max_conf < thr_by_code[predict]:
                final_code = config['other_name']
            else:
                final_code = predict
            best_bbox = select_df.iloc[0, 2]
            best_score = max_conf
        else:
            final_code = config['false_name']
            best_bbox = []
            best_score = 1

        if final_code in config['size_TRM']:
            if size < 50:
                final_code = 'TRM1'
            elif size > 150:
                final_code = 'TRMX'
            else:
                final_code = 'TRMO'
        if final_code in config['size_DRM']:
            if size < 50:
                final_code = 'DRM1'
            elif size > 150:
                final_code = 'DRMX'
            else:
                final_code = 'DRMO'

        detect_result.append(det_df)
        true_categories.append(category)
        prediction_categories.append(final_code)
        true_bounding_boxes.append(best_bbox)
        print('pred code == {},oic code={}, best_bbox is == {}, defect score == {}'.format(final_code, category, best_bbox,
                                                                                        best_score))
        log.append({'pred code': final_code, 'oic code': category, 'image name': img_name, 'defect score': best_score})

    save_confusion_matrix(true_categories, prediction_categories, label, save_file=os.path.join(out_dir_path, 'matrix_test.csv'))
    res = pd.DataFrame(log)
    res.to_csv(os.path.join(result_path, 'result_log_test.csv'), index=False)

    save_html = 'analysis_result_test.html'
    csv_file = os.path.join(result_path, 'result_log_test.csv')
    result_analysis = ResultAnalysis(csv=csv_file)
    result_analysis.main_reports(os.path.join(result_path, save_html))
    draw_bounding_boxes(true_categories, prediction_categories, true_bounding_boxes, img_lst, result_path, detect_result)


def deploy_result_final(sample_root,  deploy_path, out_dir_path, out_path=None, result_path=None, device='cuda:0', **kwargs):
    initModel(deploy_path, out_dir_path, device=device)
    img_lst = glob(sample_root + '/*/*.jpg')
    pbar = tqdm(img_lst)
    # get size
    log = []
    size = kwargs.get('size', None)
    true_categories = []
    prediction_categories = []
    true_bounding_boxes = []
    label = []
    detect_result = []
    # iter img
    for i, img_path in enumerate(pbar):
        img_name = img_path.split('/')[-1]
        category = img_path.split('/')[-2]
        img = cv2.imread(img_path)
        if img is None:
            print('img {} is None'.format(img_path))
            # continue
        result = inference_detector(model, img_path)
        json_dict = model_test(result, img_name, labels)
        det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score','area'])
        # prior parameters
        prior_lst = config['prior_order']
        if config['false_name'] not in prior_lst:
            prior_lst[config['false_name']] = len(prior_lst)
        if config['other_name'] not in prior_lst:
            prior_lst[config['other_name']] = len(prior_lst)
        thr_by_code = config['thr_by_code']
        for code in labels:
            if code not in thr_by_code:
                thr_by_code[code] = config["other_thr"]

        if len(det_df) > 0:

            max_idx = det_df.score.idxmax()
            max_conf = det_df.score[max_idx]
            predict = det_df.loc[max_idx, 'category']
            area = det_df.loc[max_idx, 'area']

            # if predict in config['confidence']:
            #     det_df_max = det_df[det_df['score'] > max_conf * 0.7]
            # else:
            det_df_max = det_df[det_df['score'] > max_conf * 0.8]
            default_conf = det_df_max.iloc[0, 3]
            for j, row in det_df_max.iterrows():
                predict_idx = prior_lst[predict]
                code = row['category']
                code_idx = prior_lst[code]
                code_area = row['area']
                code_score = row['score']
                #score above threshold by code priority,if priority same by code_socre,or
                if code_idx < predict_idx or (code_idx == predict_idx and code_score > default_conf) or (code_idx == predict_idx and code_area > area):
                    predict = code
                    area = code_area
                    default_conf = code_score

            max_conf = max(det_df.loc[det_df.category == predict, 'score'].values)
            select_df = det_df[(det_df['category'] == predict) & (det_df['score'] == max_conf)]
            if max_conf < thr_by_code[predict]:
                final_code = config['other_name']
            else:
                final_code = predict
            best_bbox = select_df.iloc[0, 2]
            best_score = max_conf
        else:
            final_code = config['false_name']
            best_bbox = []
            best_score = 1

        if final_code in config['size_TRM']:
            if size is not None:
                if size < 50:
                    final_code = 'TRM1'
                elif size >= 150:
                    final_code = 'TRMX'
                else:
                    final_code = 'TRMO'
        if final_code in config['size_DRM']:
            if size is not None:
                if size < 50:
                    final_code = 'DRM1'
                elif size >= 150:
                    final_code = 'DRMX'
                else:
                    final_code = 'DRMO'
        if final_code in config['size_MPTO']:
            if size is not None:
                if size < 100:
                    final_code = 'WPT1'

        detect_result.append(det_df)
        true_categories.append(category)
        prediction_categories.append(final_code)
        true_bounding_boxes.append(best_bbox)
        if category not in label:
            label.append(category)
        # label_bool = final_labeling(det_df, 'FLE06', best_bbox, best_score, os.path.join(out_dir_path, 'xml'),
        #                             img_path)
        print('pred code == {},oic code={}, best_bbox is == {}, defect score == {}'.format(final_code, category,
                                                                                           best_bbox,
                                                                                           best_score))
        log.append({'pred code': final_code, 'oic code': category, 'image name': img_name, 'defect score': best_score})

    save_confusion_matrix(true_categories, prediction_categories,
                          label, save_file=os.path.join(out_dir_path, 'matrix_new.csv'))
    res = pd.DataFrame(log)
    res.to_csv(os.path.join(result_path, 'result_log_new.csv'), index=False)

    save_html = 'analysis_result_new.html'
    csv_file = os.path.join(result_path, 'result_log_new.csv')
    result_analysis = ResultAnalysis(csv=csv_file)
    result_analysis.main_reports(os.path.join(result_path, save_html))
    draw_bounding_boxes(true_categories, prediction_categories, true_bounding_boxes, img_lst, result_path, detect_result)


def deploy_result_final_new(sample_root,deploy_path, out_dir_path, out_path=None, result_path=None, device='cuda:0', **kwargs):
    initModel(deploy_path, out_dir_path, device=device)
    df = pd.read_csv(os.path.join('/data/sdv2/cb/work_dir/20200804_52902/', 'test_data_0804.csv'))
    label = ['DRM1', 'DRMO', 'DRMX', 'IRP1', 'ISC3', 'MDF1', 'MPL1', 'MPLO', 'MPT3', 'MPTO', 'MRM3',
             'MSC2', 'TRM1', 'TRMO', 'TRMX', 'WPT1', 'FALSE', 'Others']
    img_lst = []
    size_list = []
    for item in df.iterrows():
        img_lst.append(item[1]['img_path'])
        size_list.append(item[1]['size'])
    pbar = tqdm(img_lst)
    # get size
    log = []
    size = kwargs.get('size', None)
    true_categories = []
    prediction_categories = []
    true_bounding_boxes = []
    detect_result = []
    # iter img
    for i, img_path in enumerate(pbar):
        img_name = img_path.split('/')[-1]
        category = img_path.split('/')[-2]
        size = size_list[i]
        img = cv2.imread(img_path)
        if img is None:
            print('img {} is None'.format(img_path))
            # continue
        result = inference_detector(model, img_path)
        json_dict = model_test(result, img_name, labels)
        det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score', 'area'])
        # prior parameters
        prior_lst = config['prior_order']
        if config['false_name'] not in prior_lst:
            prior_lst.append(config['false_name'])
        if config['other_name'] not in prior_lst:
            prior_lst.append(config['other_name'])
        thr_by_code = config['thr_by_code']
        for code in labels:
            if code not in thr_by_code:
                thr_by_code[code] = config["other_thr"]
        if len(det_df) > 0:
            # det area max
            det_df = det_df[det_df['area'] > 0.6 * det_df['area'].max()]

            max_idx = det_df.score.idxmax()
            max_conf = det_df.score[max_idx]
            predict = det_df.loc[max_idx, 'category']

            for j, row in det_df.iterrows():
                predict_idx = prior_lst.index(predict)
                code = row['category']
                code_idx = prior_lst.index(code)
                score = row['score']
                alpha = thr_by_code[code]
                if code_idx < predict_idx and score > max_conf * alpha:
                    predict = code
            max_conf = max(det_df.loc[det_df.category == predict, 'score'].values)
            select_df = det_df[(det_df['category'] == predict) & (det_df['score'] == max_conf)]
            if max_conf < 0.42:
                final_code = config['other_name']
            else:
                final_code = predict
            best_bbox = select_df.iloc[0, 2]
            best_score = max_conf
        else:
            final_code = config['false_name']
            best_bbox = []
            best_score = 1

        if final_code in config['size_TRM']:
            if size < 50:
                final_code = 'TRM1'
            elif size > 150:
                final_code = 'TRMX'
            else:
                final_code = 'TRMO'
        if final_code in config['size_DRM']:
            if size < 50:
                final_code = 'DRM1'
            elif size > 150:
                final_code = 'DRMX'
            else:
                final_code = 'DRMO'

        detect_result.append(det_df)
        true_categories.append(category)
        prediction_categories.append(final_code)
        true_bounding_boxes.append(best_bbox)
        print('pred code == {},oic code={}, best_bbox is == {}, defect score == {}'.format(final_code, category, best_bbox,
                                                                                        best_score))
        log.append({'pred code': final_code, 'oic code': category, 'image name': img_name, 'defect score': best_score})

    save_confusion_matrix(true_categories, prediction_categories, label, save_file=os.path.join(out_dir_path, 'matrix_test.csv'))
    res = pd.DataFrame(log)
    res.to_csv(os.path.join(result_path, 'result_log_test.csv'), index=False)

    save_html = 'analysis_result_test.html'
    csv_file = os.path.join(result_path, 'result_log_test.csv')
    result_analysis = ResultAnalysis(csv=csv_file)
    result_analysis.main_reports(os.path.join(result_path, save_html))
    draw_bounding_boxes(true_categories, prediction_categories, true_bounding_boxes, img_lst, result_path, detect_result)


def deploy_result_final_prior(sample_root,deploy_path, out_dir_path, out_path=None, result_path=None, device='cuda:0', **kwargs):
    initModel(deploy_path, out_dir_path, device=device)
    df = pd.read_csv(os.path.join('/data/sdv2/cb/work_dir/20200727_52902/', 'test_data_0804.csv'))
    label = ['DRM1', 'DRMO', 'DRMX', 'IRP1', 'ISC3', 'MDF1', 'MPL1', 'MPLO', 'MPT3', 'MPTO', 'MRM3',
             'MSC2', 'TRM1', 'TRMO', 'TRMX', 'WPT1', 'FALSE', 'Others']
    img_lst = []
    size_list = []
    for item in df.iterrows():
        img_lst.append(item[1]['img_path'])
        size_list.append(item[1]['size'])
    pbar = tqdm(img_lst)
    # get size
    log = []
    size = kwargs.get('size', None)
    true_categories = []
    prediction_categories = []
    true_bounding_boxes = []
    detect_result = []
    # iter img
    for i, img_path in enumerate(pbar):
        img_name = img_path.split('/')[-1]
        category = img_path.split('/')[-2]
        size = size_list[i]
        img = cv2.imread(img_path)
        if img is None:
            print('img {} is None'.format(img_path))
            # continue
        result = inference_detector(model, img_path)
        json_dict = model_test(result, img_name, labels)
        det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score', 'area'])
        # prior parameters
        prior_lst = config['prior_order']
        if config['false_name'] not in prior_lst:
            prior_lst[config['false_name']] = len(prior_lst)
        if config['other_name'] not in prior_lst:
            prior_lst[config['other_name']] = len(prior_lst)
        thr_by_code = config['thr_by_code']
        for code in labels:
            if code not in thr_by_code:
                thr_by_code[code] = config["other_thr"]
        if len(det_df) > 0:
            max_conf = 0
            best_prior_index = len(prior_lst)
            best_index = 0
            best_area = 0
            for i, row in det_df.iterrows():
                code = row['category']
                prior_index = prior_lst[code]
                if row['score'] > thr_by_code[code]:
                    if prior_index < best_prior_index or \
                            (prior_index == best_prior_index and row['area'] > best_area):
                        best_index = i
                        max_conf = row['score']
                        best_prior_index = prior_index
                        best_area = row['area']

            if max_conf == 0:
                best_index = det_df['score'].argmax()
                final_code = config['other_name']
            else:
                final_code = det_df.loc[best_index, 'category']

            best_score = det_df.loc[best_index, 'score']
            best_bbox = det_df.loc[best_index, 'bbox']
        else:
            final_code = config['false_name']
            best_bbox = []
            best_score = 1

        if final_code in config['size_TRM']:
            if size < 50:
                final_code = 'TRM1'
            elif size > 150:
                final_code = 'TRMX'
            else:
                final_code = 'TRMO'
        if final_code in config['size_DRM']:
            if size < 50:
                final_code = 'DRM1'
            elif size > 150:
                final_code = 'DRMX'
            else:
                final_code = 'DRMO'

        detect_result.append(det_df)
        true_categories.append(category)
        prediction_categories.append(final_code)
        true_bounding_boxes.append(best_bbox)
        print('pred code == {},oic code={}, best_bbox is == {}, defect score == {}'.format(final_code, category, best_bbox,
                                                                                        best_score))
        log.append({'pred code': final_code, 'oic code': category, 'image name': img_name, 'defect score': best_score})

    save_confusion_matrix(true_categories, prediction_categories, label, save_file=os.path.join(out_dir_path, 'matrix_test.csv'))
    res = pd.DataFrame(log)
    res.to_csv(os.path.join(result_path, 'result_log_test.csv'), index=False)

    save_html = 'analysis_result_test.html'
    csv_file = os.path.join(result_path, 'result_log_test.csv')
    result_analysis = ResultAnalysis(csv=csv_file)
    result_analysis.main_reports(os.path.join(result_path, save_html))
    draw_bounding_boxes(true_categories, prediction_categories, true_bounding_boxes, img_lst, result_path, detect_result)


def img_to_csv(img_dir, save_path):
    img = []
    code_list = os.listdir(img_dir)
    size_path = [i for i in code_list if i.endswith('csv')]
    size = os.path.join(img_dir, size_path[0])
    df = pd.read_csv(size)
    for row in tqdm(df.iterrows()):
        size_modul = row[1]['size']
        img_name = row[1]['img_path'].split('\\')[-1]
        nn = row[1]['manual_code']
        for code in code_list:
            if not code.endswith('csv'):
                temp_path = os.path.join(img_dir, code)
                files = os.listdir(temp_path)
                if img_name in files:
                    img.append({'m_code': code, 'cate_name': nn, 'img_path': os.path.join(temp_path, img_name),
                                'size': size_modul})
    df = pd.DataFrame(img)
    df.to_csv(save_path)


def deploy_result_final_prior_confidence(sample_root,deploy_path, out_dir_path, out_path=None, result_path=None, device='cuda:0', **kwargs):
    initModel(deploy_path, out_dir_path, device=device)
    df = pd.read_csv(os.path.join('/data/sdv2/cb/work_dir/20200824_52902', 'test_data_0824.csv'))
    label = ['DRM1', 'DRMO', 'DRMX', 'IRP1', 'MKL1', 'ISC3', 'MDF1', 'MPL1', 'MPLO', 'MPT3', 'MPTO', 'MRM3',
             'MSC2', 'TRM1', 'TRMO', 'TRMX', 'WPT1', 'FALSE', 'Others']
    img_lst = []
    size_list = []
    for item in df.iterrows():
        img_lst.append(item[1]['img_path'])
        size_list.append(item[1]['size'])
    pbar = tqdm(img_lst)
    # get size
    log = []
    size = kwargs.get('size', 0)
    sizeX = kwargs.get('sizeX', 0)
    sizeY = kwargs.get('sizeY', 0)
    unit_id = kwargs.get('unit_id', None)
    size = 0 if size is None else size
    sizeX = 0 if sizeX is None else sizeX
    sizeY = 0 if sizeY is None else sizeY
    if unit_id == 'A1AOI800':
        size = max(size, sizeX, sizeY)
    true_categories = []
    prediction_categories = []
    true_bounding_boxes = []
    detect_result = []
    # iter img
    for i, img_path in enumerate(pbar):
        img_name = img_path.split('/')[-1]
        category = img_path.split('/')[-2]
        size = size_list[i]
        img = cv2.imread(img_path)
        if img is None:
            print('img {} is None'.format(img_path))
            # continue
        result = inference_detector(model, img_path)
        json_dict = model_test(result, img_name, labels)
        det_df = pd.DataFrame(json_dict, columns=['name', 'category', 'bbox', 'score', 'bbox_score', 'area'])
        # prior parameters
        prior_lst = config['prior_order']
        if config['false_name'] not in prior_lst:
            prior_lst[config['false_name']] = len(prior_lst)
        if config['other_name'] not in prior_lst:
            prior_lst[config['other_name']] = len(prior_lst)
        thr_by_code = config['thr_by_code']
        for code in labels:
            if code not in thr_by_code:
                thr_by_code[code] = config["other_thr"]
        if len(det_df) > 0:
            max_idx = det_df.score.idxmax()
            max_conf = det_df.score[max_idx]
            predict = det_df.loc[max_idx, 'category']
            area = det_df.loc[max_idx, 'area']

            # if predict in config['confidence']:
            #     det_df_max = det_df[det_df['score'] > max_conf * 0.7]
            # else:
            det_df_max = det_df[det_df['score'] > max_conf * 0.8]
            default_conf = det_df_max.iloc[0, 3]
            for j, row in det_df_max.iterrows():
                predict_idx = prior_lst[predict]
                code = row['category']
                code_idx = prior_lst[code]
                code_area = row['area']
                code_score = row['score']
                #score above threshold by code priority,if priority same by code_socre,or
                if code_idx < predict_idx or (code_idx == predict_idx and code_score > default_conf) or (code_idx == predict_idx and code_area > area):
                    predict = code
                    area = code_area
                    default_conf = code_score

            max_conf = max(det_df.loc[det_df.category == predict, 'score'].values)
            select_df = det_df[(det_df['category'] == predict) & (det_df['score'] == max_conf)]
            if max_conf < thr_by_code[predict]:
                final_code = config['other_name']
            else:
                final_code = predict
            best_bbox = select_df.iloc[0, 2]
            best_score = max_conf
        else:
            final_code = config['false_name']
            best_bbox = []
            best_score = 1

        if final_code in config['size_TRM']:
            if size < 50:
                final_code = 'TRM1'
            elif size >= 150:
                final_code = 'TRMX'
            else:
                final_code = 'TRMO'
        if final_code in config['size_DRM']:

            if size < 50:
                final_code = 'DRM1'
            elif size >= 150:
                final_code = 'DRMX'
            else:
                final_code = 'DRMO'
        if final_code in config['size_MPTO']:
            if size < 100:
                final_code = 'WPT1'
        # if final_code in config['change_name']:
        #     final_code = config['Others']

        detect_result.append(det_df)
        true_categories.append(category)
        prediction_categories.append(final_code)
        true_bounding_boxes.append(best_bbox)
        print('pred code == {},oic code={}, best_bbox is == {}, defect score == {}'.format(final_code, category, best_bbox,
                                                                                        best_score))
        log.append({'pred code': final_code, 'oic code': category, 'image name': img_name, 'defect score': best_score})

    save_confusion_matrix(true_categories, prediction_categories, label, save_file=os.path.join(out_dir_path, 'matrix_test_{}.csv'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))))
    res = pd.DataFrame(log)
    res.to_csv(os.path.join(result_path, 'result_log_test.csv'), index=False)

    save_html = 'analysis_result_test_{}.html'.format(time.strftime('%Y-%m-%d', time.localtime()))
    csv_file = os.path.join(result_path, 'result_log_test.csv')
    result_analysis = ResultAnalysis(csv=csv_file)
    result_analysis.main_reports(os.path.join(result_path, save_html))
    draw_bounding_boxes(true_categories, prediction_categories, true_bounding_boxes, img_lst, result_path, detect_result)


if __name__ == '__main__':
    sample_root = '/data/sdv2/cb/52902_train_sample_enhance/'
    deploy_path = '/data/sdv2/cb/1x1A4_code/52902_code/'
    out_dir_path = '/data/sdv2/cb/work_dir/20200824_52902/'
    img_path = '/data/sdv2/cb/52902_test_20200806/'
    # img_path = '/data/sdv2/cb/test_sample/'
    # img_to_csv(img_path, os.path.join(out_dir_path, 'test_data_0824.csv'))
    # deploy_result_final_prior_confidence(sample_root, deploy_path, out_dir_path, result_path=out_dir_path)
    deploy_result_final(sample_root, deploy_path, out_dir_path, result_path=out_dir_path)


