from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import shutil
import os

#获取mnist数据集 并归一化
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#构建模型
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

model_save_path = "./saved_model/1"

#删除之前保存的
if os.path.isdir(model_save_path):
    shutil.rmtree(model_save_path)
os.mkdir(model_save_path)

tf.saved_model.save(model, model_save_path)