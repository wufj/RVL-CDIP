import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import glob
total_dir = 'D:/Preprocess-RVL-CDIP/images/'
image_lists = []
label_lists = []

data_dir = 'D:/Preprocess-RVL-CDIP/datasets'
tfrecord_file = data_dir + '/train/train_tif.tfrecords'
vfrecord_file = data_dir + '/valid/valid_tif.tfrecords'

def compose(dataset):
    with open('./labels/' +dataset +".txt") as fh:
        sets = (fh.read().split('\n'))
        # print(sets)
        dt = {}

        for row in sets:
            try:
                kv = row.split(' ')
                key = kv[0]
                val = kv[1]
                image_lists.append(key)
                label_lists.append(int(val))
                print(key)
                # train_filenames = tf.constant([train_dir + key])
                # train_labels = tf.constant([val])
                # print(train_filenames)
                # print(train_labels)
            except:
                print('whoops')
        # print(len(dt))

        return image_lists,label_lists

#生成tfrecode文件
def write_record(tfrecord_file,files,lablels):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for filename, label in zip(files, lablels):
            image = open(filename, 'rb').read()     # 读取数据集图片到内存，image 为一个 Byte 类型的字符串
            feature = {                             # 建立 tf.train.Feature 字典
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # 图片是一个 Bytes 对象
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))   # 标签是一个 Int 对象
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature)) # 通过字典建立 Example
            writer.write(example.SerializeToString())   # 将Example序列化并写入 TFRecord 文件
if __name__ == '__main__':
    image_lists,label_lists = compose('val')
    image_lists =[total_dir + filename for filename in image_lists]
    label_lists = [filename for filename in label_lists]
    # write_record(tfrecord_file,image_lists,label_lists)
    write_record(vfrecord_file, image_lists, label_lists)