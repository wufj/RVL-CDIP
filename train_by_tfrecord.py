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
num_epochs = 10
path_train = 'dataset1/train/*/*.jpg'
path_val = 'dataset1/val/*/*.jpg'
data_dir = 'D:/Preprocess-RVL-CDIP/datasets'
tfrecord_file = data_dir + '/train/train_shuffle.tfrecords'
vfrecord_file = data_dir + '/valid/valid_shuffle.tfrecords'
BATCH_SIZE = 64


def dealData(path):
    imags_path = glob.glob(path)  # 全局获取文件的路径
    all_label_names = [image_p.split('\\')[1] for image_p in imags_path]  # 获取所有图片的label
    label_names = np.unique(all_label_names)  # 去重
    label_to_index = dict((name, i) for i, name in enumerate(label_names))  # 类和数字形成字典
    index_to_label = dict((v, k) for k, v in label_to_index.items())  # 类和数字的字典键值对反转
    all_labels = [int(label_to_index.get(name)) for name in all_label_names]  # 获取所有图片对应数字的类
    # imags_path = tf.constant([filename for filename in imags_path])
    # all_labels = tf.constant([filename for filename in all_labels])
    imags_path = [filename for filename in imags_path]
    all_labels = [filename for filename in all_labels]
    return imags_path, all_labels

# 我们可以通过以下代码，读取之前建立的 train.tfrecords 文件，并通过 Dataset.map 方法，使用 tf.io.parse_single_example 函数对数据集中的每一个序列化的 tf.train.Example 对象解码。
def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])    # 解码JPEG图片
    feature_dict['image']  = tf.image.resize(feature_dict['image'], [256, 256]) / 255.0
    return feature_dict['image'], feature_dict['label']

def read_record(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件
    dataset = raw_dataset.map(_parse_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label


def pic_show(vfrecord_file):
    dataset1 = read_record(vfrecord_file)
    import matplotlib.pyplot as plt
    for image, label in dataset1:
        print(label)
        plt.title('cat' if label == 0 else 'dog')
        plt.imshow(image.numpy())
        plt.show()

train_dataset = read_record(tfrecord_file)
vali_dataset = read_record(vfrecord_file)
# print(train_dataset['image'])

# 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
# train_dataset = train_dataset.shuffle(buffer_size=2300)
train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# vali_dataset = vali_dataset.shuffle(buffer_size=2300)
vali_dataset = vali_dataset.batch(32)
vali_dataset = vali_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 1)),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Conv2D(32, 5, activation='relu'),
#     tf.keras.layers.MaxPooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(16, activation='softmax')
#  ])
model = tf.keras.applications.VGG16(weights=None, input_shape=(256, 256, 1),classes=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

# model.fit(train_dataset, epochs=num_epochs)

#

train_path, train_labels = dealData(path_train)
val_path, val_labels = dealData(path_val)


train_count = len(train_path)
val_count = len(val_path)


steps_per_epoch = train_count // BATCH_SIZE
validation_steps = val_count // BATCH_SIZE

# model.fit(train_dataset, epochs=num_epochs)
#
# print(steps_per_epoch)
# print(validation_steps)
history = model.fit(train_dataset, epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=vali_dataset,
                    validation_steps=validation_steps)




# pic_show(tfrecord_file)