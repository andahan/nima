import numpy as np
import os
import tensorflow as tf
import pandas as pd



def parse_data(filename, scores):
    '''
    Loads the image file, and randomly applies crops and flips to each image.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (256, 256))
    image = tf.random_crop(image, size=(IMAGE_SIZE, IMAGE_SIZE, 3))
    image = tf.image.random_flip_left_right(image)
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def parse_data_without_augmentation(filename, scores):
    '''
    Loads the image file without any augmentation. Used for validation set.

    Args:
        filename: the filename from the record
        scores: the scores from the record

    Returns:
        an image referred to by the filename and its scores
    '''
    image = tf.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = (tf.cast(image, tf.float32) - 127.5) / 127.5
    return image, scores

def train_generator(batchsize, shuffle=True):
    '''
    Creates a python generator that loads the AVA dataset images with random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for training
        shuffle: whether to shuffle the dataset

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        # create a dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_scores))
        train_dataset = train_dataset.map(parse_data, num_parallel_calls=2)
        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=4)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)

def val_generator(batchsize):
    '''
    Creates a python generator that loads the AVA dataset images without random data
    augmentation and generates numpy arrays to feed into the Keras model for training.

    Args:
        batchsize: batchsize for validation set

    Returns:
        a batch of samples (X_images, y_scores)
    '''
    with tf.Session() as sess:
        val_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_scores))
        val_dataset = val_dataset.map(parse_data_without_augmentation)
        val_dataset = val_dataset.batch(batchsize)
        val_dataset = val_dataset.repeat()
        val_iterator = val_dataset.make_initializable_iterator()

        val_batch = val_iterator.get_next()

        sess.run(val_iterator.initializer)

        while True:
            try:
                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)
            except:
                val_iterator = val_dataset.make_initializable_iterator()
                sess.run(val_iterator.initializer)
                val_batch = val_iterator.get_next()

                X_batch, y_batch = sess.run(val_batch)
                yield (X_batch, y_batch)

def features_generator(record_path, faeture_size, batchsize, shuffle=True):
    '''
    Creates a python generator that loads pre-extracted features from a model
    and serves it to Keras for pre-training.

    Args:
        record_path: path to the TF Record file
        faeture_size: the number of features in each record. Depends on the base model.
        batchsize: batchsize for training
        shuffle: whether to shuffle the records

    Returns:
        a batch of samples (X_features, y_scores)
    '''
    with tf.Session() as sess:
        # maps record examples to numpy arrays

        def parse_single_record(serialized_example):
            # parse a single record
            example = tf.parse_single_example(
                serialized_example,
                features={
                    'features': tf.FixedLenFeature([faeture_size], tf.float32),
                    'scores': tf.FixedLenFeature([10], tf.float32),
                })

            features = example['features']
            scores = example['scores']
            return features, scores

        # Loads the TF dataset
        train_dataset = tf.data.TFRecordDataset([record_path])
        train_dataset = train_dataset.map(parse_single_record, num_parallel_calls=4)

        train_dataset = train_dataset.batch(batchsize)
        train_dataset = train_dataset.repeat()
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=5)
        train_iterator = train_dataset.make_initializable_iterator()

        train_batch = train_iterator.get_next()

        sess.run(train_iterator.initializer)

        # indefinitely extract batches
        while True:
            try:
                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)
            except:
                train_iterator = train_dataset.make_initializable_iterator()
                sess.run(train_iterator.initializer)
                train_batch = train_iterator.get_next()

                X_batch, y_batch = sess.run(train_batch)
                yield (X_batch, y_batch)




def load_data_and_labels(sample_path,base_images_path):
    """
    high label[1,0,0], middle [0,1,0],low [0,0,1]
    Returns images paths and labels.
    """
    # Load data from files
    df = pd.read_csv(sample_path, encoding='utf-8', sep=',')
    # print(len(df))
    # df = df.drop_duplicates(df["feedid"])
    # print(len(df))

    # Generate imgs
    img_paths = []
    for index, pic_name in enumerate(df["feedid"]):
        img_path = base_images_path + str(pic_name) + '.jpg'  #pic以feedid命名
        if os.path.exists(img_path)==0:
            df = df.drop(index)
            continue
        else:
            img_paths.append(img_path)

    # Generate labels
    def transform_one_hot(labels):
        n_labels = 3
        one_hot = np.eye(n_labels)[labels]
        return one_hot

    labels = transform_one_hot(df["manual_quality"]-1)

    return np.array(img_paths), labels

# test:
# base_images_path = './images/'
# train_sample_path = 'train1.txt'
# test_sample_path = 'test1.txt'

base_images_path = '/data/ceph_11015/ssd/jerrycen/video_tonality/data/'
train_sample_path = '/data/ceph_11015/ssd/jerrycen/video_tonality/nima/train.txt'
test_sample_path = '/data/ceph_11015/ssd/jerrycen/video_tonality/nima/test.txt'

IMAGE_SIZE = 224

train_image_paths,train_scores = load_data_and_labels(train_sample_path,base_images_path)
test_image_paths,test_scores = load_data_and_labels(test_sample_path,base_images_path)

print('Train set size : ', train_image_paths.shape, train_scores.shape)
print('Val set size : ', test_image_paths.shape, test_scores.shape)
print('Train and validation datasets ready !')