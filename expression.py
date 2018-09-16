import tensorflow as tf
import cv2
from config import EXPRESSION_TENSORS_PATH
from utils import load_tensors, format_image
from os.path import exists, join
from os import makedirs
import json


class ExpressionEstimator():
    pass


def expression_nn(inputs):
    x = tf.reshape(inputs, shape=[-1, 48, 48, 1])
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=64,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        name='expression_conv1')
    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=[2, 2],
        padding='same',
        name='expression_pool1')
    
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=[1, 1],
        padding='same',
        activation=tf.nn.relu,
        name='expression_conv2')
    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[3, 3],
        strides=[2, 2],
        padding='same',
        name='expression_pool2')
    
    norm2 = tf.nn.lrn(
        input=pool2,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75)
    
    dense1 = tf.layers.dense(
        inputs=tf.reshape(norm2, [-1, 12 * 12 * 64]),
        units=384,
        activation=tf.nn.relu,
        name='expression_dense1'
    )
    
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=192,
        name='expression_dense2'
    
    )
    
    logits = tf.layers.dense(
        inputs=dense2,
        units=7,
        name='expression_logits'
    )
    return logits


def build_graph():
    inputs = tf.placeholder(shape=[48, 48], dtype=tf.float32)
    
    logits = expression_nn(inputs)
    probs = tf.nn.softmax(logits)
    
    for v in tf.trainable_variables():
        print(v)
    
    tensors = load_tensors(EXPRESSION_TENSORS_PATH)
    print(tensors.keys())
    
    sess = tf.Session()
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(v)
        v.load(tensors[v.name], sess)
    
    return sess, [probs, logits, inputs]


# print('logits', sess.run(logits))
# result = sess.run()


# ret, frame = video_captor.read()
# frame = cv2.imread('surprise.png')


def get_expression_result(image, output_path=None, output_name=None):
    if not exists(output_path):
        makedirs(output_path)
    
    detected_face, face_edge = format_image(image)
    print('Face', detected_face, 'Edge', face_edge)
    if face_edge is not None:
        [x, y, w, h] = face_edge
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if detected_face is not None:
        cv2.imwrite(join(output_path, output_name + '.expression.png'), detected_face)
        p, l = sess.run([probs, logits], feed_dict={inputs: detected_face})
        print(p, l)
        with open(join(output_path, output_name + '.emotion.p.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(p.tolist()))
        with open(join(output_path, output_name + '.emotion.l.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(l.tolist()))


if __name__ == '__main__':
    image = cv2.imread('test.png')
    output_path = 'test'
    output_name = 'test'
    get_expression_result(image, output_path, output_name)
