import tensorflow as tf
import os

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = 'D:/tensorflow_study/datasets/fastai-datasets-cats-vs-dogs-2'
train_cats_dir = data_dir + '/train/cats/'
train_dogs_dir = data_dir + '/train/dogs/'
test_cats_dir = data_dir + '/valid/cats/'
test_dogs_dir = data_dir + '/valid/dogs/'
train_dir = 'D:/Preprocess-RVL-CDIP/images/'
train_list = []
label_list = []

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
                train_list.append(key)
                label_list.append(int(val))
                print(key)
                # train_filenames = tf.constant([train_dir + key])
                # train_labels = tf.constant([val])
                # print(train_filenames)
                # print(train_labels)
            except:
                print('whoops')
        # print(len(dt))

        return train_list,label_list

def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)            # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_resized = tf.image.resize(image_decoded, [256, 256]) / 255.0
    return image_resized, label

if __name__ == '__main__':
    train_list,label_list = compose('val')
    train_filenames = tf.constant([train_dir+filename for filename in train_list])
    train_labels = tf.constant([filename for filename in label_list])
    # train_labels = train_labels.astype(tf.int32)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))

    train_dataset = train_dataset.map(
        map_func=_decode_and_resize,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # 取出前buffer_size个数据放入buffer，并从其中随机采样，采样后的数据用后续数据替换
    train_dataset = train_dataset.shuffle(buffer_size=23000)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy]
    )

    model.fit(train_dataset, epochs=num_epochs)