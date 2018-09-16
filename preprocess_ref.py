import cv2
from os.path import join, exists, isdir
from os import listdir, makedirs
from shutil import move
import re
import numpy as np
import json
from multiprocessing import Pool
import tensorflow as tf

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

with open('/private/var/py/ReadHeart/20180830.data.sr.score.1581.json', encoding='utf-8') as f:
    data = json.loads(f.read())

len(data)

len(data['keys'])

input_size = 128


def get_video_file(key):
    file_template = '/private/var/py/ReadHeart/static/datas/{key}/{key}.mp4'.format(key=key)
    return file_template


def process(input_key):
    if exists('pretrain/' + input_key):
        return
    
    # input_key = data['keys']
    input_file = get_video_file(input_key)
    
    print('Input file', input_file)
    
    stream = cv2.VideoCapture(input_file)
    
    mark_detector = MarkDetector()
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]
    
    frame = None
    count = 0
    frame_got = True
    
    target_folder = 'pretrain/{key}'.format(key=input_key)
    
    if not exists(target_folder):
        makedirs(target_folder)
    
    while frame_got:
        frame_got, frame = stream.read()
        if frame_got:
            # print('Frame', frame)
            facebox = mark_detector.extract_cnn_facebox(frame)
            # print('Facebox', facebox)
            
            if facebox:
                
                height, width = frame.shape[:2]
                pose_estimator = PoseEstimator(img_size=(height, width))
                face_img = frame[facebox[1]: facebox[3],
                           facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (input_size, input_size))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                marks = mark_detector.detect_marks(face_img)
                
                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]
                
                # print('Marks', marks)
                
                for mark in marks:
                    cv2.circle(frame, tuple(mark), 3, (255, 0, 0))
                
                # print('Marks Length', len(marks))
                
                pose = pose_estimator.solve_pose_by_68_points(marks)
                # print('Pose', pose)
                pose = [pose[0].tolist(), pose[1].tolist()]
                with open('{target_folder}/vecs.{count}.txt'.format(count=count, target_folder=target_folder),
                          'w') as f:
                    f.write(json.dumps(pose))
                with open('{target_folder}/marks.{count}.txt'.format(count=count, target_folder=target_folder),
                          'w') as f:
                    f.write(json.dumps(marks.tolist()))
                cv2.imwrite('{target_folder}/image.{count}.png'.format(count=count, target_folder=target_folder), frame)
            count += 1


# for input_key in input_keys:


pool = Pool()
input_keys = data['keys']
pool.map(process, input_keys)
