import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import ipywidgets as ipw
import numpy as np
import matplotlib.pylab as plt
from ipywidgets import interactive
from PIL import Image
import os


def variable_summaries(name, var):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def img_summaries(name, var):
    with tf.name_scope(name):
        tf.summary.image(name, var)


def conv33132_summaries(name, conv_filter):
    assert conv_filter.shape.as_list()[3] == 32
    assert conv_filter.shape.as_list()[0] == 3
    assert conv_filter.shape.as_list()[1] == 3

    conv_images = tf.split(conv_filter, 32, 3)  # 32 x [3, 3, 1, 1]
    row0 = tf.concat(conv_images[0:8], 0)  # [24, 3, 1, 1]
    row1 = tf.concat(conv_images[8:16], 0)
    row2 = tf.concat(conv_images[16:24], 0)
    row3 = tf.concat(conv_images[24:32], 0)
    conv_image = tf.concat([row0, row1, row2, row3], 1)  # [24, 12, 1, 1]
    conv_image = tf.reshape(conv_image, [1, 24, 12, 1])
    img_summaries(name, conv_image)


def conv55124_summary(name, conv_filter):
    assert conv_filter.shape.as_list()[3] == 24
    assert conv_filter.shape.as_list()[0] == 5
    assert conv_filter.shape.as_list()[1] == 5

    padding = tf.zeros([5, 5, 1, 1])  # [5, 5, 1, 1]
    conv_filter_padded = tf.concat([conv_filter, padding], 3)  # [5, 5, 1, 25]
    conv_images = tf.split(conv_filter_padded, 25, 3)        # 25 x [5, 5, 1, 1]
    row0 = tf.concat(conv_images[0:5], 0)   # [25, 5, 1, 1]
    row1 = tf.concat(conv_images[5:10], 0)  # [25, 5, 1, 1]
    row2 = tf.concat(conv_images[10:15], 0)  # [25, 5, 1, 1]
    row3 = tf.concat(conv_images[15:20], 0)  # [25, 5, 1, 1]
    row4 = tf.concat(conv_images[20:25], 0)  # [25, 5, 1, 1]
    conv_image = tf.concat([row0, row1, row2, row3, row4], 1)  # [25, 25, 1, 1]
    conv_image = tf.reshape(conv_image, [1, 25, 25, 1])
    img_summaries(name, conv_image)


def get_sprite_img(images, img_shape):
    image_cout = len(images)
    h, w = img_shape[:2]

    rows = int(np.ceil(np.sqrt(image_cout)))
    cols = rows

    if len(img_shape) == 3:
        sprite_img = np.zeros([rows * h, cols * w, img_shape[2]])
    else:
        sprite_img = np.zeros([rows * h, cols * w])

    image_id = 0
    for row_id in range(rows):
        for col_id in range(cols):
            if image_id >= image_cout:
                if len(img_shape) == 3:
                    sprite_img = Image.fromarray(np.uint8(sprite_img))
                else:
                    sprite_img = Image.fromarray(np.uint8(sprite_img * 0xFF))
                return sprite_img

            row_pos = row_id * h
            col_pos = col_id * w
            sprite_img[row_pos:row_pos + h, col_pos:col_pos + w] = images[image_id].reshape(img_shape)
            image_id += 1


def get_label_class_names(label_class_onehots, class_id2class_name_mapping):
    return [class_id2class_name_mapping[c_id] for c_id in np.argmax(label_class_onehots, axis=1).tolist()]


def save_label_class_names(label_class_names, path):
    with open(path, 'w') as fw:
        for name in label_class_names:
            fw.write(name + '\n')


def init_embedding_projector(file_writer, embedding, img_shape):
    projector_config = projector.ProjectorConfig()
    embedding_config = projector_config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.metadata_path = 'label_class_names.tsv'
    embedding_config.sprite.image_path = 'sprite_img.png'
    embedding_config.sprite.single_image_dim.extend((img_shape[0], img_shape[1]))
    projector.visualize_embeddings(file_writer, projector_config)


def init_embedding_data(file_writer_dir, sprite_img, label_names):
    sprite_img_file = 'sprite_img.png'
    label_class_names_file = 'label_class_names.tsv'

    sprite_img_path = os.path.join(file_writer_dir, sprite_img_file)
    label_class_names_path = os.path.join(file_writer_dir, label_class_names_file)

    save_label_class_names(label_names, label_class_names_path)
    sprite_img.save(sprite_img_path)


def get_conv_filter_control_params(shape, init_mode='diag_pos'):
    controls_params = {}
    for row_id in range(shape[0]):
        for col_id in range(shape[1]):
            pos_id = str(row_id) + ", " + str(col_id)

            if init_mode == 'diag_right':
                default_value = -1 if row_id + col_id < max(list(shape)) else 1
            elif init_mode == 'diag_left':
                default_value = 1 if row_id + col_id < max(list(shape)) else -1
            elif init_mode == 'top_half':
                default_value = 1 if row_id < shape[0] / 2 else -1
            elif init_mode == 'bottom_half':
                default_value = -1 if row_id < shape[0] / 2 else 1
            elif init_mode == 'top':
                default_value = 1 if row_id == 0 else -1
            elif init_mode == 'bottom':
                default_value = 1 if row_id == shape[0] - 1 else -1
            elif init_mode == 'left':
                default_value = 1 if col_id == 0 else -1
            elif init_mode == 'right':
                default_value = 1 if col_id == shape[1] - 1 else -1
            else:
                default_value = 0

            controls_params[pos_id] = ipw.BoundedFloatText(
                value=default_value, min=-1, max=1, step=0.1, description="",
                layout=ipw.Layout(width='200px', height='200px', margin='0px 0px 0px 0px', padding='0px 0px 0px 0px'))
    return controls_params


def get_conv_filter_control_widget(widgets_children, shape):
    controls_matrix = list()
    idx = 0
    for row_id in range(shape[0]):
        controls_matrix.append(list())
        for col_id in range(shape[1]):
            widgets_children[idx].description = ''
            controls_matrix[row_id].append(widgets_children[idx])
            idx += 1
    return ipw.VBox([ipw.HBox(m, layout=ipw.Layout(align_items='stretch')) for m in controls_matrix],
                    layout=ipw.Layout(width='430px', height='430px', padding='15px 0px 0px 30px'))


def conv_filter_widget(image, conv_filter_shape=(4, 4), stride_row=1, stride_col=1, init_mode='diag_pos'):
    conv_filter_shape[0] = min(6, conv_filter_shape[0])
    conv_filter_shape[1] = min(6, conv_filter_shape[1])

    def plot_conv_filter(**kwargs):
        conv_filter = np.zeros(conv_filter_shape, dtype=np.float32)

        for row_id in range(conv_filter_shape[0]):
            for col_id in range(conv_filter_shape[1]):
                pos_id = str(row_id) + ", " + str(col_id)
                conv_filter[row_id, col_id] = kwargs[pos_id]

        conv_filter_reshaped = conv_filter.reshape(conv_filter_shape[0], conv_filter_shape[1], 1, 1)
        image_reshaped = image.reshape((1, 28, 28, 1))

        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            tf_image = tf.constant(image_reshaped)
            tf_conv_filter = tf.constant(conv_filter_reshaped)
            tf_conv_layer = tf.nn.conv2d(tf_image, tf_conv_filter,
                                         strides=[1, stride_row, stride_col, 1], padding="VALID")

        with tf.Session(graph=graph) as sess:
            conv_layer = sess.run(tf_conv_layer)[0]

        conv_layer_shape = (np.floor((28 - conv_filter_shape[0] + stride_row) / stride_row).astype(int),
                            np.floor((28 - conv_filter_shape[1] + stride_col) / stride_col).astype(int))

        fig = plt.figure(figsize=(15, 15))
        plt.subplots_adjust(left=0, right=0.9, top=0.9, bottom=0.1)

        ax = plt.subplot(1, 2, 1)
        ax.axes.set_axis_off()
        plt.title('convolution filter')
        im = ax.imshow(conv_filter)
        plt.colorbar(im, orientation='horizontal')

        ax = plt.subplot(1, 2, 2)
        ax.axes.set_axis_off()
        im = ax.imshow(conv_layer.reshape(conv_layer_shape))
        plt.title('convolution layer')
        plt.colorbar(im, orientation='horizontal')
        plt.show()

    conv_filter_controls_params = get_conv_filter_control_params(conv_filter_shape, init_mode)
    widget = interactive(plot_conv_filter, **conv_filter_controls_params)
    return ipw.VBox([get_conv_filter_control_widget(widget.children[:-1], conv_filter_shape),
                     widget.children[-1]], layout=ipw.Layout(margin='50px 30px 50px 50px'))
