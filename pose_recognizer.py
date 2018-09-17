import cv2
import tensorflow as tf
import json
from config import IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, LANDMARK_TENSORS_PATH, VIDEO
from pose_estimator import PoseEstimator
from utils import load_tensors, format_image_rgb, check_dir
import numpy as np
from os.path import join, exists
from os import makedirs


class PoseRecognizer():
    
    def __init__(self):
        self.build_graph()
        self.build_model()
    
    def _landmark_nn(self, inputs):
        """
        landmark network
        :param inputs: image inputs
        :return: landmark_logits
        """
        inputs = tf.reshape(inputs, [-1, IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, 3])
        
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
    
    def build_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(shape=[IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, 3],
                                         dtype=tf.float32)
            self.landmark_logits = self._landmark_nn(self.inputs)
    
    def build_model(self):
        tensors = load_tensors(LANDMARK_TENSORS_PATH)
        print(tensors.keys())
        with self.graph.as_default():
            self.sess = tf.Session()
            for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                print(v)
                v.load(tensors[v.name], self.sess)
    
    def process(self, image, output_path=None, output_name=None):
        image_origin = image
        check_dir(output_path)
        
        cv2.imwrite(join(output_path, output_name + '.origin.png'), image_origin)
        # image_name =
        # image_origin = cv2.imread(image_path)
        image_origin_height, image_origin_width = image_origin.shape[0:2]
        # print('Width', image_origin_width, 'Height', image_origin_height)
        
        image_crop, image_edge = format_image_rgb(image_origin)
        cv2.imwrite(join(output_path, output_name + '.landmark.crop.png'), image_crop)
        # print('Image Data', image_crop、, 'Image Edge', image_edge)
        
        image_crop_resize = cv2.resize(image_crop, (128, 128))
        cv2.imwrite(join(output_path, output_name + '.landmark.resize.png'), image_crop_resize)
        
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        # print('Image', image_crop_resize)
        
        predictions = self.sess.run(self.landmark_logits, feed_dict={self.inputs: image_crop_resize})
        # print(predictions)
        # print('Len predictions', predictions)
        
        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))
        # print(marks)
        
        # width =
        # print('Image edge shape', image_edge)
        # to do multiply
        marks *= (image_edge[2] - image_edge[0])
        marks[:, 0] += image_edge[0]
        marks[:, 1] += image_edge[1]
        # print(marks)
        
        with open(join(output_path, output_name + '.marks.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(marks.tolist()))
        
        for mark in marks:
            cv2.circle(image_origin, tuple(mark), 3, (255, 0, 0))
        
        cv2.imwrite(join(output_path, output_name + '.landmark.png'), image_origin)
        
        pose_estimator = PoseEstimator(img_size=(image_origin_height, image_origin_width))
        # pose_estimator
        pose = pose_estimator.solve_pose_by_68_points(marks)
        print('Pose', pose)
        
        with open(join(output_path, output_name + '.pose.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(pose))
        
        return pose
    
    # def get_head_poses(self, video, output_path):
    #     video = cv2.VideoCapture(VIDEO)
    #
    #     flag = True
    #
    #     count = 1
    #
    #     poses = []
    #
    #     while flag:
    #         flag, frame = video.read()
    #         print(frame)
    #         # cv2.imwrite(join('video_ouput1'))
    #         try:
    #             pose = get_head_pose(frame, output_path, '%s.png' % count)
    #             poses.append(pose)
    #         except:
    #             pass
    #         count += 1
    #
    #     result = np.std(np.asarray(poses), axis=0).tolist()
    #     with open(join(output_path, 'poses.txt'), 'w', encoding='utf-8') as f:
    #         f.write(json.dumps(result))
    #     return result


# get_head_pose()
#
# result = get_head_poses(VIDEO, 'video_output3')
# print(result)

if __name__ == '__main__':
    pr = PoseRecognizer()
    file = 'test.png'
    image = cv2.imread(file)
    output_path = 'test'
    output_name = 'test'
    pr.process(image, output_path, output_name)
