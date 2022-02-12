'''
使用onnx模型做前向推理
不需要再定义模型结构
'''
import cv2
import numpy as np
from tqdm import tqdm
import core.utils as utils
from core.config import cfg
import onnxruntime as rt

class yolo3_onnx:
    def __init__(self, model_path):
        self.build_model(model_path)

    def build_model(self, model):
        self._model = rt.InferenceSession(model)
        self._input_name  = self._model.get_inputs()[0].name  
        self._output_name = self._model.get_outputs()[0].name

    def detect(self, np_img):
        image_size = np_img.shape[:2]
        image_data = utils.image_preporcess(np.copy(np_img), [cfg.DETECT.INPUT_SIZE, cfg.DETECT.INPUT_SIZE])
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self._model.run([], {self._input_name:image_data})
        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
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

            # cv2.imshow("detected", cv2.resize(image_rst,(0,0),fx=0.25,fy=0.25))
            # cv2.waitKey(0)

            save_path = file_[:file_.rfind('.')]+'_drawed_v3_onnx.jpg'
            cv2.imwrite(save_path, image_rst)


def main():
    detector = yolo3_onnx('weights/yolov3.onnx')
    detector.detect_batch()

if __name__ == '__main__':
    main()