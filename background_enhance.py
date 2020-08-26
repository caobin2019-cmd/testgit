import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.etree import ElementTree as aa
from tqdm import tqdm
import random
import string


class Creat_new_background(object):

    def __init__(self, background_path, foreground_path_with_xml, save_path):
        self.background = background_path
        self.foreground = foreground_path_with_xml
        self.save_path = save_path

    def crop_with_xml(self):
        crop_info = []
        code_lists = os.listdir(self.foreground)
        for code in code_lists:
            tem_path = os.path.join(self.foreground, code)
            total_files = os.listdir(tem_path)
            for file in total_files:
                if file.endswith('jpg'):
                    xml_name = file.split('.')[0] + '.xml'
                    xml_path = os.path.join(tem_path, xml_name)
                    img = cv2.imread(os.path.join(tem_path, file))
                    info_img = self.parse_xml_bbox(xml_path)
                    object = info_img['object']
                    width, height = info_img['width'], info_img['height']
                    for bbox in object:
                        roi = img[bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
                        label = bbox[4]
                        crop_info.append({'img_crop': roi, 'gt_label': label, 'relate_xml': xml_path,
                                          'width': width, 'height': height, 'crop_width': int(bbox[2]-bbox[0]),
                                          'crop_height': int(bbox[3]-bbox[1])})
        return crop_info

    def parse_xml_bbox(self, xml_path):
        img_info = {}
        img_info['object'] = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for ele in root.iter():
            if 'width' in ele.tag:
                img_info['width'] = int(ele.text)
            if 'height' in ele.tag:
                img_info['height'] = int(ele.text)
            if 'object' in ele.tag:
                object_info = [0, 0, 0, 0, 0]
                for attr in list(ele):
                    if 'name' in attr.tag:
                        label = attr.text
                        object_info[4] = label
                    if 'bndbox' in attr.tag:
                        for pos in list(attr):
                            if 'xmin' in pos.tag:
                                object_info[0] = int(pos.text)
                            if 'ymin' in pos.tag:
                                object_info[1] = int(pos.text)
                            if 'xmax' in pos.tag:
                                object_info[2] = int(pos.text)
                            if 'ymax' in pos.tag:
                                object_info[3] = int(pos.text)
                img_info['object'].append(object_info)
        return img_info

    def create_new_img_random(self, crop_info):
        background_files = os.listdir(self.background)
        length = len(crop_info)
        for background_file in tqdm(background_files):
            random_data = np.random.randint(0, length, 1)[0]
            crop =crop_info[random_data]
            info = {}
            guid = self.generate_random_str(6)
            back_path = os.path.join(self.background, background_file)
            back_xml_name = background_file.split('.')[0] + guid + '.xml'
            info['path'] = os.path.join(self.background, background_file.split('.')[0] + guid + '.jpg')
            info['filename'] = background_file.split('.')[0] + guid + '.jpg'
            info['xml_name'] = back_xml_name
            img = cv2.imread(back_path)
            h, w = img.shape[:2]
            width = crop['width']
            height = crop['height']
            gt_label = crop['gt_label']
            roi = crop['img_crop']
            crop_width = crop['crop_width']
            crop_height = crop['crop_height']
            info['folder'] = gt_label
            info['width'] = w
            info['height'] = h

            if h == height and w == width:
                random_x = np.random.randint(0, w-crop_width, 1)
                random_y = np.random.randint(0, h-crop_height, 1)
                random_x = random_x[0]
                random_y = random_y[0]
                bbox = [random_x, random_y, random_x+crop_width, random_y+crop_height]
                info['bndbox'] = bbox
                img[random_y:random_y+roi.shape[0], random_x:random_x+roi.shape[1]] = roi # fill
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                cv2.imwrite(os.path.join(self.save_path, info['filename']), img)
                self.creat_annotation(info)

            else:
                ratio_h = h/height
                ratio_w = w/width
                true_crop_height = int(ratio_h * crop_height)
                true_crop_width = int(ratio_w * crop_width)
                roi_true = cv2.resize(roi, (true_crop_height, true_crop_width))
                random_x = np.random.randint(0, w - true_crop_width, 1)
                random_y = np.random.randint(0, h - true_crop_height, 1)
                random_x = random_x[0]
                random_y = random_y[0]
                bbox = [random_x, random_y, random_x + true_crop_width, random_y + true_crop_height]
                info['bndbox'] = bbox
                img[random_x:random_x + true_crop_width, random_y:random_y + true_crop_height] = roi_true  # fill
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                cv2.imwrite(os.path.join(self.save_path, info['filename']), img)
                self.creat_annotation(info)

    def create_new_img(self, crop_info):
        background_files = os.listdir(self.background)
        print('len={}'.format(len(background_files)))
        for crop in tqdm(crop_info):
            for background_file in tqdm(background_files):
                info = {}
                guid = self.generate_random_str(6)
                back_path = os.path.join(self.background, background_file)
                back_xml_name = background_file.split('.')[0] + guid + '.xml'
                info['path'] = os.path.join(self.background, background_file.split('.')[0] + guid + '.jpg')
                info['filename'] = background_file.split('.')[0] + guid + '.jpg'
                info['xml_name'] = back_xml_name
                img = cv2.imread(back_path)
                h, w = img.shape[:2]
                width = crop['width']
                height = crop['height']
                gt_label = crop['gt_label']
                roi = crop['img_crop']
                crop_width = crop['crop_width']
                crop_height = crop['crop_height']
                info['folder'] = gt_label
                info['width'] = w
                info['height'] = h

                if h == height and w == width:
                    random_x = np.random.randint(0, w-crop_width, 1)
                    random_y = np.random.randint(0, h-crop_height, 1)
                    random_x = random_x[0]
                    random_y = random_y[0]
                    bbox = [random_x, random_y, random_x+crop_width, random_y+crop_height]
                    info['bndbox'] = bbox
                    img[random_x:random_x+crop_width+1, random_y:random_y+crop_height+1] = roi # fill
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    cv2.imwrite(os.path.join(self.save_path, info['filename']), img)
                    self.creat_annotation(info)

                else:
                    ratio_h = h/height
                    ratio_w = w/width
                    true_crop_height = int(ratio_h * crop_height)
                    true_crop_width = int(ratio_w * crop_width)
                    roi_true = cv2.resize(roi, (true_crop_height, true_crop_width))
                    random_x = np.random.randint(0, w - true_crop_width, 1)
                    random_y = np.random.randint(0, h - true_crop_height, 1)
                    random_x = random_x[0]
                    random_y = random_y[0]
                    bbox = [random_x, random_y, random_x + true_crop_width, random_y + true_crop_height]
                    info['bndbox'] = bbox
                    img[random_x:random_x + true_crop_width + 1, random_y:random_y + true_crop_height + 1] = roi_true  # fill
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    cv2.imwrite(os.path.join(self.save_path, info['filename']), img)
                    self.creat_annotation(info)

    def create_new_img_without_background(self, crop_info):
        background_files = os.listdir(self.background)
        print('len={}'.format(len(background_files)))
        for crop in tqdm(crop_info):
            for background_file in tqdm(background_files):
                info = {}
                guid = self.generate_random_str(6)
                back_path = os.path.join(self.background, background_file)
                back_xml_name = background_file.split('.')[0] + guid + '.xml'
                info['path'] = os.path.join(self.background, background_file.split('.')[0] + guid + '.jpg')
                info['filename'] = background_file.split('.')[0] + guid + '.jpg'
                info['xml_name'] = back_xml_name
                img = cv2.imread(back_path)
                h, w = img.shape[:2]
                width = crop['width']
                height = crop['height']
                gt_label = crop['gt_label']
                roi = crop['img_crop']
                crop_width = crop['crop_width']
                crop_height = crop['crop_height']
                info['folder'] = gt_label
                info['width'] = w
                info['height'] = h
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                mask_inv = cv2.bitwise_not(mask)
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                if h == height and w == width:
                    random_x = np.random.randint(0, w-crop_width, 1)
                    random_y = np.random.randint(0, h-crop_height, 1)
                    random_x = random_x[0]
                    random_y = random_y[0]
                    bbox = [random_x, random_y, random_x+crop_width, random_y+crop_height]
                    info['bndbox'] = bbox

                    img_roi = img[random_x:random_x+crop_width+1, random_y:random_y+crop_height+1]
                    print(img_roi.shape)
                    img_fg = cv2.bitwise_and(img_roi, img_roi, mask=mask)
                    dst = cv2.add(roi_bg, img_fg)
                    img[random_x:random_x+crop_width+1, random_y:random_y+crop_height+1] = dst # fill

                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    cv2.imwrite(os.path.join(self.save_path, info['filename']), img)
                    self.creat_annotation(info)

                else:
                    ratio_h = h/height
                    ratio_w = w/width
                    true_crop_height = int(ratio_h * crop_height)
                    true_crop_width = int(ratio_w * crop_width)
                    roi_true = cv2.resize(roi, (true_crop_height, true_crop_width))
                    # handle
                    roi_gray = cv2.cvtColor(roi_true, cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    mask_inv = cv2.bitwise_not(mask)
                    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                    random_x = np.random.randint(0, w - true_crop_width, 1)
                    random_y = np.random.randint(0, h - true_crop_height, 1)
                    random_x = random_x[0]
                    random_y = random_y[0]
                    bbox = [random_x, random_y, random_x + true_crop_width, random_y + true_crop_height]
                    info['bndbox'] = bbox

                    img_roi = img[random_x:random_x + true_crop_width + 1, random_y:random_y + true_crop_height + 1]
                    img_fg = cv2.bitwise_and(img_roi, img_roi, mask=mask)

                    dst = cv2.add(roi_bg, img_fg)
                    img[random_x:random_x + true_crop_width + 1, random_y:random_y + true_crop_height + 1] = roi_true  # fill
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    cv2.imwrite(os.path.join(self.save_path, info['filename']), img)
                    self.creat_annotation(info)

    def creat_annotation(self, img_info):
        root = aa.Element('annotation')
        folder = aa.SubElement(root, 'folder')
        folder.text = img_info['folder']
        filename = aa.SubElement(root, 'filename')
        filename.text = img_info['filename']
        path = aa.SubElement(root, 'path')
        path.text = img_info['path']
        source = aa.SubElement(root, 'source')
        database = aa.SubElement(source, 'database')
        database.text = 'UnKnown'
        size = aa.SubElement(root, 'size')
        width = aa.SubElement(size, 'width')
        width.text = str(img_info['width'])
        height = aa.SubElement(size, 'height')
        height.text = str(img_info['height'])
        depth = aa.SubElement(size, 'depth')
        depth.text = '3'
        segmented = aa.SubElement(root, 'segmented')
        segmented.text = '0'
        object = aa.SubElement(root, 'object')
        name = aa.SubElement(object, 'name')
        name.text = img_info['folder']
        pose = aa.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = aa.SubElement(object, 'truncated')
        truncated.text = '0'
        difficult = aa.SubElement(object, 'difficult')
        difficult.text = '0'
        bndbox = aa.SubElement(object, 'bndbox')
        xmin = aa.SubElement(bndbox, 'xmin')
        xmin.text = str(img_info['bndbox'][0])
        ymin = aa.SubElement(bndbox, 'ymin')
        ymin.text = str(img_info['bndbox'][1])
        xmax = aa.SubElement(bndbox, 'xmax')
        xmax.text = str(img_info['bndbox'][2])
        ymax = aa.SubElement(bndbox, 'ymax')
        ymax.text = str(img_info['bndbox'][3])
        tree = aa.ElementTree(root)
        tree.write(os.path.join(self.save_path, img_info['xml_name']), encoding='utf-8', xml_declaration=True)

    def generate_random_str(self, randomlength=6):
        str_list = [random.choice(string.digits + string.ascii_letters) for i in range(randomlength)]
        random_str = ''.join(str_list)
        return random_str

    def create_without_bbox(self, back_img, detect):
        '''
        用于目标区域去轮廓提取，需要框选的目前背景比较干净
        :param back_img:
        :param detect:
        :return:
        '''
        h, w = detect.shape[:2]
        random_x = np.random.randint(0, back_img.shape[1]-w, 1)[0]
        random_y = np.random.randint(0, back_img.shape[0]-h, 1)[0]
        roi = back_img[random_y:random_y+h, random_x:random_x+w]
        detect_gray = cv2.cvtColor(detect, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(detect_gray, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        mask_inv = cv2.bitwise_not(mask)
        img_bg = cv2.bitwise_and(detect, detect, mask=mask_inv)
        roi_fg = cv2.bitwise_and(roi, roi, mask=mask)
        dst = cv2.add(img_bg, roi_fg)
        back_img[random_y:random_y + h, random_x:random_x + w] = dst
        return back_img



if __name__ == '__main__':
    f_path = r'C:\Users\78\Desktop\test'
    b_path = r'C:\Users\78\Desktop\www\False'
    save_path = r'C:\Users\78\Desktop\www\img_MPTO'

    l = Creat_new_background(b_path, f_path, save_path)
    info = l.crop_with_xml()
    l.create_new_img_random(info)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # for crop in info:
    #     cv2.imshow('img', crop['img_crop'])
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # for crop in info:
    #     for img_path in os.listdir(b_path):
    #         fa = cv2.imread(os.path.join(b_path, img_path))
    #
    #         img = crop['img_crop']
    #         roi = fa[100:100+img.shape[0], 100:100+img.shape[1]]
    #         print(roi.shape)
    #         img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #         mask_inv = cv2.bitwise_not(mask)
    #         img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    #         roi_fg = cv2.bitwise_and(roi, roi, mask=mask)
    #         dst = cv2.add(img_bg, roi_fg)
    #         fa[100:100+img.shape[0], 100:100+img.shape[1]] = dst
    #         cv2.imshow('img', fa)
    #         cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # #


