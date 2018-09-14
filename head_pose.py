import cv2
import tensorflow as tf
import json
from config import IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, LANDMARK_TENSORS_PATH

image = cv2.imread('test.png')
print(image)

inputs = tf.placeholder(shape=[1, IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, 3], dtype=tf.float32)





def landmark_nn(inputs):
    """
    landmark network
    :param inputs: image inputs
    :return: landmark_logits
    """
    # layer 1
    conv1 = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv1')
    
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid',
        name='landmark_pool1')
    
    # layer2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv2')
    
    conv3 = tf.layers.conv2d(
        inputs=conv2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv3')
    
    pool2 = tf.layers.max_pooling2d(
        inputs=conv3,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid',
        name='landmark_pool2')
    
    # layer3
    conv4 = tf.layers.conv2d(
        inputs=pool2,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv4')
    
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=64,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv5')
    
    pool3 = tf.layers.max_pooling2d(
        inputs=conv5,
        pool_size=[2, 2],
        strides=(2, 2),
        padding='valid',
        name='landmark_pool3')
    
    # layer4
    conv6 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv6')
    
    conv7 = tf.layers.conv2d(
        inputs=conv6,
        filters=128,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv7')
    
    pool4 = tf.layers.max_pooling2d(
        inputs=conv7,
        pool_size=[2, 2],
        strides=(1, 1),
        padding='valid',
        name='landmark_pool4')
    
    # layer5
    conv8 = tf.layers.conv2d(
        inputs=pool4,
        filters=256,
        kernel_size=[3, 3],
        strides=(1, 1),
        padding='valid',
        activation=tf.nn.relu,
        name='landmark_conv8')
    
    flatten = tf.layers.flatten(inputs=conv8)
    
    # layer6
    dense = tf.layers.dense(
        inputs=flatten,
        units=1024,
        activation=tf.nn.relu,
        use_bias=True,
        name='landmark_dense')
    
    # logits
    logits = tf.layers.dense(
        inputs=dense,
        units=136,
        activation=None,
        use_bias=True,
        name='landmark_logits')
    return logits


landmark_logits = landmark_nn(inputs)

for v in tf.trainable_variables():
    print(v)

tensors = load_tensors(LANDMARK_TENSORS_PATH)
print(tensors.keys())

sess = tf.Session()
for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    print(v)
    v.load(tensors[v.name], sess)
    
    