- 第三方 yolov3 的 tensorflow 实现，包含训练和检测模块  
https://yunyang1994.github.io/posts/YOLOv3

# 训练  

1. 创建如下目录树  
|--VOCdevkit  
|--|--VOC2019  
|--|--|--Annotations  
|--|--|--ImageSets  
|--|--|--JPEGImages  
|--|--|--labels  

2. 将所有待训练的图片放到JPEGImages下，将labelImg输出的所有标注结果（xml）放到Annotations下  

3. 使用tools/xml2label.py生成训练文件  
// image path | tl_x,tl_y,br_x,br_y,label | ... 
/VOCdevkit/VOC2019/JPEGImages/IMG_5811.JPG 2038,466,3264,2448,0 1600,700,2105,1671,0 

4. 指定预训练权重文件路径  
train.py 42行  
'''
model.load_weights("./weights/yolov3_agv")
'''
也可以不指定，从0开始训练  

5. 修改配置参数  
5.1 __C.YOLO.IOU_LOSS_THRESH：越小对框的惩罚越大  
5.2 __C.TRAIN.LR_END: 越大学习率的下限越高  

6. python train.py 开始训练  
tensorboard --logdir ./data/log 开启可视化  

---

# 检测

1. 修改config.py内 __C.DETECT 相关的参数  

2. python detect.py 开始检测
