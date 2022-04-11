#! /usr/bin/env python
# coding=utf-8

from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

# Set the class name
__C.YOLO.CLASSES              = "./data/classes/agv.names"
__C.YOLO.ANCHORS              = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.6

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.WEIGHT_FILE         = "./weights/yolov4/yolov4.weights"
__C.TRAIN.ANNOT_PATH          = "./data/dataset/agv_train.txt"
__C.TRAIN.BATCH_SIZE          = 4
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-5
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS  = 10 
__C.TRAIN.SECOND_STAGE_EPOCHS = 40



# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/yymnist_test.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.3
__C.TEST.IOU_THRESHOLD        = 0.45


# DETECT options
__C.DETECT                    = edict()

__C.DETECT.FILE_LIST          = "./data/dataset/imgfile.list"
__C.DETECT.INPUT_SIZE         = 416
__C.DETECT.WEIGHT_PATH        = "./weights/yolov4"
__C.DETECT.SCORE_THRESHOLD    = 0.33
__C.DETECT.IOU_THRESHOLD      = 0.4

