import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov4 import YOLOv4, decode, compute_loss
from core.config import cfg

trainset = Dataset('train')
logdir   = "./data/log"

steps_per_epoch      = len(trainset)
global_steps         = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps         = cfg.TRAIN.WARMUP_EPOCHS*steps_per_epoch
first_stage_epoches  = cfg.TRAIN.FISRT_STAGE_EPOCHS
second_stage_epoches = cfg.TRAIN.SECOND_STAGE_EPOCHS
total_steps          = (first_stage_epoches + second_stage_epoches)*steps_per_epoch
freeze_layers        = utils.load_freeze_layer()


input_tensor = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
conv_tensors = YOLOv4(input_tensor)

output_tensors = []
for i, conv_tensor in enumerate(conv_tensors):
    pred_tensor = decode(conv_tensor, i)
    output_tensors.append(conv_tensor)
    output_tensors.append(pred_tensor)

model = tf.keras.Model(input_tensor, output_tensors)
# model.load_weights(cfg.TRAIN.WEIGHT_FILE)
utils.load_weights(model, cfg.TRAIN.WEIGHT_FILE)

optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir): shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = pred_result[i*2], pred_result[i*2+1]
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP: %4d/%4d lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))

        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5*(cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                 (1+tf.cos((global_steps-warmup_steps) / (total_steps-warmup_steps) * np.pi)))
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


isfreeze = False
for epoch in range(first_stage_epoches + second_stage_epoches):
    if epoch < first_stage_epoches:
        if not isfreeze:
            isfreeze = True
            for lname in freeze_layers:
                layer_freeze = model.get_layer(lname)
                utils.freeze_all(layer_freeze)
    elif isfreeze:
        isfreeze = False
        for lname in freeze_layers:
            layer_freeze = model.get_layer(lname)
            utils.unfreeze_all(layer_freeze)

    for image_data , target in trainset:
        train_step(image_data, target)
    model.save_weights("./weights/yolov4")