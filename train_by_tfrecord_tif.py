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
BATCH_SIZE = 64

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


# 我们可以通过以下代码，读取之前建立的 train.tfrecords 文件，并通过 Dataset.map 方法，使用 tf.io.parse_single_example 函数对数据集中的每一个序列化的 tf.train.Example 对象解码。
def _parse_example(example_string): # 将 TFRecord 文件中的每一个序列化的 tf.train.Example 解码
    feature_description = { # 定义Feature结构，告诉解码器每个Feature的类型是什么
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.image.decode_image(feature_dict['image'],channels=3)    # 解码JPEG图片
    feature_dict['image']  = tf.cast(feature_dict['image'] , tf.float32)
    feature_dict['image'] = tf.image.resize_images(feature_dict['image'], [256, 256]) / 255.0
    # feature_dict['image'] = tf.reshape(feature_dict['image'] , tf.stack([256, 256, 3]) )
    # feature_dict['image'].set_shape([256, 256, 3])
    # feature_dict['image'] =  feature_dict['image']/tf.uint8(255)
    return feature_dict['image'], feature_dict['label']

def read_record(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)    # 读取 TFRecord 文件
    dataset = raw_dataset.map(_parse_example,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

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

train_dataset = train_dataset.batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# vali_dataset = vali_dataset.shuffle(buffer_size=2300)
vali_dataset = vali_dataset.batch(32)
vali_dataset = vali_dataset.prefetch(tf.data.experimental.AUTOTUNE)


model = tf.keras.applications.VGG16(weights=None, input_shape=(256, 256, 3),classes=16)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=[tf.keras.metrics.sparse_categorical_accuracy]
)

train_path, train_labels = compose('train')
val_path, val_labels = compose('val')

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