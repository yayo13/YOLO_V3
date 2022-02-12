'''
使用权重模型（checkpoint）做前向推理
需要定义模型结构
'''
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov3 import YOLOv3, decode

class yolo3_tf:
    def __init__(self):
        self.build_model()

    def build_model(self):
        input_layer = tf.keras.layers.Input([cfg.DETECT.INPUT_SIZE, cfg.DETECT.INPUT_SIZE, 3])
        output_layer = YOLOv3(input_layer)

        bbox_tensors = []
        for i, fm in enumerate(output_layer):
            bbox_tensor = decode(fm, i)
            bbox_tensors.append(bbox_tensor)

        self.model = tf.keras.Model(input_layer, bbox_tensors)
        self.model.load_weights(cfg.DETECT.WEIGHT_PATH)

        ##### frozen model #####
        import onnx
        import tf2onnx
        # self.model.save('pb/yolov3.h5')
        model_onnx, _ = tf2onnx.convert.from_keras(self.model)
        onnx.save_model(model_onnx, 'weights/yolov3.onnx')

    def detect(self, np_img):
        image_size = np_img.shape[:2]
        image_data = utils.image_preporcess(np.copy(np_img), [cfg.DETECT.INPUT_SIZE, cfg.DETECT.INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, cfg.DETECT.INPUT_SIZE, cfg.DETECT.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.DETECT.IOU_THRESHOLD, method='nms')

        image_drawed = utils.draw_bbox(np_img, bboxes)
        return image_drawed

    def detect_batch(self):
        with open(cfg.DETECT.FILE_LIST) as f:
            lines = f.readlines()
            image_files = [line.strip() for line in lines if len(line.strip()) != 0]
        
        for file_ in tqdm(image_files):
            image = cv2.imread(file_)
            image_rst = self.detect(image)

            save_path = file_[:file_.rfind('.')]+'_drawed_v3.jpg'
            cv2.imwrite(save_path, image_rst)


def main():
    detector = yolo3_tf()
    # detector.detect_batch()

if __name__ == '__main__':
    main()