import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import glob
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout

path_train = 'dataset1/train/*/*.jpg'
path_val = 'dataset1/val/*/*.jpg'
data_dir = 'D:/Preprocess-RVL-CDIP/datasets'
tfrecord_file = data_dir + '/train/train_shuffle.tfrecords'
vfrecord_file = data_dir + '/valid/valid_shuffle.tfrecords'
from PIL import Image

def is_valid(file):
    valid = True
    try:
        Image.open(file).load()
        # print(file)
    except OSError:
        valid = False
        print(file)
    return valid


def dealData(path):
    imags_path = glob.glob(path)  # 全局获取文件的路径
    # for image in imags_path:
    #     valid = is_valid(image)
        # if(valid == False):
        #      imags_path.remove(image)

    all_label_names = [image_p.split('\\')[1] for image_p in imags_path]  # 获取所有图片的label
    label_names = np.unique(all_label_names)  # 去重
    label_to_index = dict((name, i) for i, name in enumerate(label_names))  # 类和数字形成字典
    index_to_label = dict((v, k) for k, v in label_to_index.items())  # 类和数字的字典键值对反转
    all_labels = [int(label_to_index.get(name)) for name in all_label_names]  # 获取所有图片对应数字的类
    # imags_path = tf.constant([filename for filename in imags_path])
    # all_labels = tf.constant([filename for filename in all_labels])
    imags_path = [filename for filename in imags_path]
    all_labels = [filename for filename in all_labels]
    print('imags_path',len(imags_path))
    print('all_labels',len(all_labels))
    index = np.random.permutation(len(imags_path))

    # 图片和路径同批次产生随机
    imags_path = np.array(imags_path)[index]
    all_labels = np.array(all_labels)[index]

    # imags_path = np.array(imags_path)
    # all_labels = np.array(all_labels)
    #
    # imags_path = imags_path.numpy()[index]
    # all_labels = all_labels.numpy()[index]

    return imags_path, all_labels

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


# 我们可以通过以下代码，读取之前建立的 train.tfrecords 文件，并通过 Dataset.map 方法，使用 tf.io.parse_single_example 函数对数据集中的每一个序列化的 tf.train.Example 对象解码。
def _parse_example(example_string):  # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_description = {  # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])  # 解码JPEG图片
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']


def read_record(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)  # 读取 TFRecord 文件
    dataset = raw_dataset.map(_parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset




# pic_show(vfrecord_file)

train_path, train_labels = dealData(path_train)
# val_path, val_labels = dealData(path_val)
write_record(tfrecord_file,train_path,train_labels)
# write_record(vfrecord_file,val_path,val_labels)