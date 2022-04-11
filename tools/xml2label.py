import xml.etree.ElementTree as ET
from tqdm import tqdm
import os

IMAGE_DIR  = '/Users/mayuan/Desktop/YOLO_V1/VOCdevkit/VOC2019/JPEGImages'
XML_DIR    = '/Users/mayuan/Desktop/YOLO_V1/VOCdevkit/VOC2019/Annotations'
LABEL_FILE = '/Users/mayuan/Desktop/YOLO_V1/VOCdevkit/VOC2019/agv_train.txt'

classes = ["agv"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(file_path):
    if not os.path.exists(file_path): return ()

    with open(file_path) as annot:
        tree=ET.parse(annot)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        objs = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            # b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            # objs.append((convert((w,h), b), cls_id))
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
            objs.append((b, cls_id))
    return objs

def main():
    image_files = [os.path.join(IMAGE_DIR, file_) for file_ in os.listdir(IMAGE_DIR)]

    with open(LABEL_FILE, 'w') as label_:
        for image_file in tqdm(image_files):
            annot_file = os.path.basename(image_file)
            annot_file = os.path.join(XML_DIR ,annot_file[:annot_file.rfind('.')]+'.xml')
            annot_info = convert_annotation(annot_file)
            if len(annot_info) == 0: continue

            line = image_file
            for obj in annot_info:
                line += ' %d,%d,%d,%d,%d'%(obj[0][0],obj[0][1],obj[0][2],obj[0][3],obj[1])
            label_.write(line+'\n')
        label_.close()

if __name__ == '__main__':
    main()

