#!/data/anaconda3/bin/python
# -*- coding: utf-8 -*-
# from pip._internal import main
# main(["install","keras"])


import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K
from data_loader import train_generator, val_generator

# Parameters
# ==================================================
#定义数据源路径
# Data loading params
tf.flags.DEFINE_string("h5_path", "/data/ceph_11015/ssd/jerrycen/video_tonality/nima/weights/mobilenet_weights.h5", "h5 path")
# tf.flags.DEFINE_string("h5_path", "./weights/mobilenet_weights.h5", "h5 path")

# Model Hyperparameters
tf.flags.DEFINE_integer("labels_dim", 3, "Dimensionality of labels (default: 3)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.75, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_float("lr", 1e-3, "learning rate (default: 1e-3)")

# Misc Parameters
# allow_soft_placement=软约束,就是运算设备不存在系统指定一个
# log_device_placement=记录设备信息,被指派另一个,要不要记录下来信息
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS #FLAGS保存命令行参数的数据

class TensorBoardBatch(TensorBoard):
    def __init__(self, *args, **kwargs):
        super(TensorBoardBatch, self).__init__(*args)
        self.tf = __import__('tensorflow')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)
        self.writer.flush()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = self.tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch * self.batch_size)
        self.writer.flush()

def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)


# ■■■■■■■■ [2]模型设计 ■■■■■■■■
image_size = 224
# ———— base模型
base_model = MobileNet((image_size, image_size, 3), alpha=1, weights=None, include_top=False, pooling='avg')

for layer in base_model.layers:
    layer.trainable = False # 让不想参加训练的层[冻结].

# ———— 主模型
x = Dropout(FLAGS.dropout_keep_prob)(base_model.output)
x = Dense(10, activation='softmax')(x)
model_main = Model(base_model.input, x)
model_main.load_weights(FLAGS.h5_path ,by_name=True)

checkpoint = ModelCheckpoint(FLAGS.h5_path, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True,mode='min')
tensorboard = TensorBoardBatch()
callbacks = [checkpoint, tensorboard]

# ———— (fine-tune构建)提取'dense_1'层的输出 ——————
layer_name = 'dense_1'
intermediate_layer_model = Model(input=base_model.input,
                                 output=model_main.get_layer(layer_name).output)
outputs_inter = Dense(FLAGS.labels_dim, activation='softmax')(intermediate_layer_model.output)
model_inter = Model(input=base_model.input, output=outputs_inter)
model = model_inter
model.summary()

# ■■■■■■■■ [3]模型编译 ■■■■■■■■
# 编译，loss function，训练过程中计算准确率
model.compile(optimizer=Adam(lr=FLAGS.lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ■■■■■■■■ [4]训练模型 ■■■■■■■■
train_generator = train_generator(batchsize=FLAGS.batch_size)
validation_generator = val_generator(batchsize=FLAGS.batch_size)
model.fit_generator(train_generator,
                    steps_per_epoch=(150000. // FLAGS.batch_size),
                    epochs=FLAGS.epochs, verbose=1, callbacks=callbacks,
                    validation_data = validation_generator,
                    validation_steps=(50000. // FLAGS.batch_size))