import cv2

import tensorflow as tf

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

CNN_INPUT_SIZE = 128


# import tensorflow.contrib.eager as tfe

# tfe.enable_eager_execution()

def load_model():
    mark_model = '/var/py/head-pose-estimation/assets/frozen_inference_graph.pb'
    
    # Get a TensorFlow session ready to do landmark detection
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    
    # for v in detection_graph.node:
    #     print('Variable', v)
    
    # print('Variables', [n.name for n in tf.get_default_graph().as_graph_def().node])
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(mark_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    graph = detection_graph
    sess = tf.Session(graph=detection_graph)
    print(graph)
    
    var_map = {}
    
    for op in graph.get_operations():
        print(str(op.name))
        
        try:
            temp = graph.get_tensor_by_name(str(op.name) + ":0")
            # print(sess.run(temp))
            result = sess.run(temp)
            var_map[op.name] = result
        except:
            print('Error', op.name)
    
    print(len(var_map))
    # print(var_map.keys())
    return var_map


var_map = load_model()

# inputs = tf.to_float(features['x'], name="input_to_float")
inputs = tf.placeholder(shape=[1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3], dtype=tf.float32)

# |== Layer 1 ==|

# Convolutional layer.
# Computes 32 features using a 3x3 filter with ReLU activation.
conv1 = tf.layers.conv2d(
    inputs=inputs,
    filters=32,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Pooling layer.
# First max pooling layer with a 2x2 filter and stride of 2.
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,
    pool_size=[2, 2],
    strides=(2, 2),
    padding='valid')

# |== Layer 2 ==|

# Convolutional layer
# Computes 64 features using a 3x3 filter with ReLU activation.
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Convolutional layer
# Computes 64 features using a 3x3 filter with ReLU activation.
conv3 = tf.layers.conv2d(
    inputs=conv2,
    filters=64,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Pooling layer
# Second max pooling layer with a 2x2 filter and stride of 2.
pool2 = tf.layers.max_pooling2d(
    inputs=conv3,
    pool_size=[2, 2],
    strides=(2, 2),
    padding='valid')

# |== Layer 3 ==|

# Convolutional layer
# Computes 64 features using a 3x3 filter with ReLU activation.
conv4 = tf.layers.conv2d(
    inputs=pool2,
    filters=64,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Convolutional layer
# Computes 64 features using a 3x3 filter with ReLU activation.
conv5 = tf.layers.conv2d(
    inputs=conv4,
    filters=64,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Pooling layer
# Third max pooling layer with a 2x2 filter and stride of 2.
pool3 = tf.layers.max_pooling2d(
    inputs=conv5,
    pool_size=[2, 2],
    strides=(2, 2),
    padding='valid')

# |== Layer 4 ==|

# Convolutional layer
# Computes 128 features using a 3x3 filter with ReLU activation.
conv6 = tf.layers.conv2d(
    inputs=pool3,
    filters=128,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Convolutional layer
# Conputes 128 features using a 3x3 filter with ReLU activation.
conv7 = tf.layers.conv2d(
    inputs=conv6,
    filters=128,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# Pooling layer
# Fourth max pooling layer with a 2x2 filter and stride of 2.
pool4 = tf.layers.max_pooling2d(
    inputs=conv7,
    pool_size=[2, 2],
    strides=(1, 1),
    padding='valid')

# |== Layer 5 ==|

# Convolutional layer
conv8 = tf.layers.conv2d(
    inputs=pool4,
    filters=256,
    kernel_size=[3, 3],
    strides=(1, 1),
    padding='valid',
    activation=tf.nn.relu)

# |== Layer 6 ==|

# Flatten tensor into a batch of vectors
flatten = tf.layers.flatten(inputs=conv8)

# Dense layer 1, a fully connected layer.
dense1 = tf.layers.dense(
    inputs=flatten,
    units=1024,
    activation=tf.nn.relu,
    use_bias=True)

# Dense layer 2, also known as the output layer.
logits = tf.layers.dense(
    inputs=dense1,
    units=136,
    activation=None,
    use_bias=True,
    name="logits")

# print(tf.all_variables())

sess = tf.Session()
INPUT_FILE = '/private/var/py/ReadHeart/static/datas/1532335528.0497806/1532335528.0497806.mp4'

# cam = cv2.VideoCapture(video_src)
cam = cv2.VideoCapture(INPUT_FILE)
# _, sample_frame = cam.read()
#
#
# print('sample_frame', sample_frame)
#
# # Introduce mark_detector to detect landmarks.
mark_detector = MarkDetector()
# facebox = mark_detector.extract_cnn_facebox(sample_frame)
# print('Facebox', facebox)
#
# height, width = sample_frame.shape[:2]
# pose_estimator = PoseEstimator(img_size=(height, width))

# Introduce scalar stabilizers for pose.
pose_stabilizers = [Stabilizer(
    state_num=2,
    measure_num=1,
    cov_process=0.1,
    cov_measure=0.1) for _ in range(6)]
# sess =
frame = None
count = 0

import numpy as np

frame_got = True
import json

with open('/private/var/py/AttitudeDetection/head_pose_tensors.json', encoding='utf-8') as f:
    data = json.loads(f.read())

while frame_got:
    frame_got, frame = cam.read()
    
    print('Frame', frame)
    facebox = mark_detector.extract_cnn_facebox(frame)
    print('Facebox', facebox)
    
    height, width = frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))
    
    # cv2.line(frame, (50, 30), (450, 35), (255, 0, 0), thickness=5)
    
    face_img = frame[facebox[1]: facebox[3],
               facebox[0]: facebox[2]]
    face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # marks = mark_detector.detect_marks(face_img)
    # print('Marks', marks)
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print('=' * 30)
        print(v)
        print(v.name)
        print('=' * 30)
        v.load(data[v.name], sess)
    
    print('FFFFace', face_img)
    
    predictions = sess.run(
        logits,
        feed_dict={inputs: [face_img]})
    
    # Convert predictions to landmarks.
    marks = np.array(predictions).flatten()
    marks = np.reshape(marks, (-1, 2))
    
    # Convert the marks locations from local CNN to global image.
    marks *= (facebox[2] - facebox[0])
    marks[:, 0] += facebox[0]
    marks[:, 1] += facebox[1]
    
    print('Marks', marks)
    
    for mark in marks:
        cv2.circle(frame, tuple(mark), 3, (255, 0, 0))
    
    cv2.imwrite('preview.png', frame)
    
    print('Marks Length', len(marks))
    # Uncomment following line to show raw marks.
    # mark_detector.draw_marks(
    #     frame, marks, color=(0, 255, 0))
    
    # Try pose estimation with 68 points.
    pose = pose_estimator.solve_pose_by_68_points(marks)
    print('Pose', pose)
    with open('preview5/image{number}.txt'.format(number=count), 'w') as f:
        f.write(str(pose))
    cv2.imwrite('preview5/image{number}.png'.format(number=count), frame)
    count += 1
    break

# #
# while count != 45:
#     frame_got, frame = cam.read()
#     count += 1
# print('Shape', frame.shape)
# print('Frame', frame)
# if frame_got is False:
#     break

# Crop it if frame is larger than expected.
# frame = frame[0:480, 300:940]

# If frame comes from webcam, flip it so it looks like a mirror.
# if video_src == 0:
#     frame = cv2.flip(frame, 2)
#
# print('Frame', frame)
# facebox = mark_detector.extract_cnn_facebox(frame)
# print('Facebox', facebox)
#
# height, width = frame.shape[:2]
# pose_estimator = PoseEstimator(img_size=(height, width))
#
#
# # cv2.line(frame, (50, 30), (450, 35), (255, 0, 0), thickness=5)
#
#
# face_img = frame[facebox[1]: facebox[3],
#            facebox[0]: facebox[2]]
# face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
# face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
# marks = mark_detector.detect_marks(face_img)
#
# # Convert the marks locations from local CNN to global image.
# marks *= (facebox[2] - facebox[0])
# marks[:, 0] += facebox[0]
# marks[:, 1] += facebox[1]
#
# print('Marks', marks)
#
#
# for mark in marks:
#     cv2.circle(frame, tuple(mark), 3, (255, 0, 0))
#
# cv2.imwrite('preview.png', frame)
#
#
# print('Marks Length', len(marks))
# # Uncomment following line to show raw marks.
# # mark_detector.draw_marks(
# #     frame, marks, color=(0, 255, 0))
#
# # Try pose estimation with 68 points.
# pose = pose_estimator.solve_pose_by_68_points(marks)
# print('Pose', pose)
