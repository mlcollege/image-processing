import pandas as pd
import numpy as np
import os
import shutil
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf


def init_dir(base_dir, target_name='', remove_existing=False):
    result_dir = os.path.join(base_dir, target_name)
    if os.path.isdir(result_dir):
        if remove_existing:
            shutil.rmtree(result_dir)
            os.makedirs(result_dir)
    else:
        os.makedirs(result_dir)
    return result_dir


def init_model_logging(base_dir, experiment, graph, remove_existing=True):
    experiment_dir = init_dir(base_dir, experiment, remove_existing=remove_existing)
    valid_writer_dir = init_dir(experiment_dir, 'valid', remove_existing=remove_existing)
    train_writer_dir = init_dir(experiment_dir, 'train', remove_existing=remove_existing)
    model_path = os.path.join(valid_writer_dir, 'model.ckpt')

    with graph.as_default():
        saver = tf.train.Saver()

    valid_writer = tf.summary.FileWriter(valid_writer_dir, graph)
    train_writer = tf.summary.FileWriter(train_writer_dir, graph)

    logging_meta = {
        'valid_writer': valid_writer,
        'valid_writer_dir': valid_writer_dir,
        'train_writer': train_writer,
        'train_writer_dir': train_writer_dir,
        'saver': saver,
        'model_path': model_path
    }

    return logging_meta


def get_image_from_url(url, image_shape=[224, 224, 3], reshape=True):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert('RGB').resize((image_shape[1], image_shape[0]))
    else:
        raise AttributeError("Wrong url")
    if reshape:
        return np.array(img).reshape([image_shape[0] * image_shape[1] * image_shape[2]])
    else:
        return np.array(img)


class Dataset(object):
    def __init__(self, dataframe, data_path_prefix, image_shape=[224, 224, 3], one_hot=False, norm=False, reshape=True):
        self.data = pd.read_hdf(dataframe, 'data')
        self.data = self.data.sample(frac=1)
        self.data_path_prefix = data_path_prefix
        self.image_shape = image_shape
        self.one_hot = one_hot
        self.norm = norm
        self.reshape = reshape

    @property
    def images(self):
        return self._get_image_batch(self.data)

    @property
    def labels(self):
        if not self.one_hot:
            return np.argmax(np.array(self.data['labels'].values.tolist()), axis=1)
        return np.array(self.data['labels'].values.tolist())

    def next_batch(self, batch_size):
        data_sample = self.data.sample(batch_size)
        if not self.one_hot:
            return self._get_image_batch(data_sample), np.argmax(np.array(data_sample['labels'].values.tolist()), axis=1)
        return self._get_image_batch(data_sample), np.array(data_sample['labels'].values.tolist())

    def _get_image_batch(self, df):
        imgs = []
        for img in df.images:
            img_path = os.path.join(self.data_path_prefix, img)
            np_img = np.array(Image.open(open(img_path, 'rb')).convert('RGB').resize((self.image_shape[1], self.image_shape[0])))
            np_img = np_img.reshape(self.image_shape[0] * self.image_shape[1] * 3)
            if self.norm:
                np_img = np_img / 255.
            imgs.append(np_img)
        stacked_imgs = np.vstack(imgs)
        if self.reshape==False:
            return stacked_imgs.reshape([-1]+self.image_shape)
        else:
            return stacked_imgs
