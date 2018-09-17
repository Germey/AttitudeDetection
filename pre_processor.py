from config import VIDEO_PATH_TEMPLATE, KEYS_FILE
import json
from expression_recognizer import ExpressionRecognizer
from pose_recognizer import PoseRecognizer
import cv2
from os.path import join, exists
from utils import check_dir
import numpy as np
from multiprocessing import Pool
from copy import deepcopy


class PreProcessor():
    
    def __init__(self):
        self.er = ExpressionRecognizer()
        self.pr = PoseRecognizer()
        self.output_path = './items'
    
    def get_keys_and_srs(self):
        with open(KEYS_FILE, encoding='utf-8') as f:
            data = json.loads(f.read())
            return data['keys'], data['sr_score_pairs']
    
    def get_video_path(self, key):
        path = VIDEO_PATH_TEMPLATE.format(key=key)
        return path
    
    def process_video(self, key):
        
        if exists(join(self.output_path, key)):
            print('Item', key, 'exists, passed')
            return
            
            # keys = self.get_keys()
        video_path = self.get_video_path(key)
        video = cv2.VideoCapture(video_path)
        # video = cv2.VideoCapture(VIDEO)
        
        flag = True
        count = 1
        poses = []
        while flag:
            flag, frame = video.read()
            print(frame)
            if not flag:
                break
            try:
                # cv2.imwrite(join('video_ouput1'))
                self.er.process(deepcopy(frame), join(self.output_path, key, str(count)), 'frame')
                pose = self.pr.process(deepcopy(frame), join(self.output_path, key, str(count)), 'frame')
                poses.append(pose)
            except:
                pass
            count += 1
        
        result = np.std(np.asarray(poses), axis=0).tolist()
        check_dir(join(self.output_path, key))
        with open(join(self.output_path, key, 'poses.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(result))
        return result
    
    def process_sr(self, key, sr):
        with open(join(self.output_path, key, 'sr.txt'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(sr))
    
    def process_all(self):
        keys, srs = self.get_keys_and_srs()
        for key, sr in zip(keys, srs):
            self.process_video(key)
            self.process_sr(key, sr)
            # break
        # print(keys, srs)
        # pool = Pool()
        # pool.map(self.process_video, keys)


#


# with open('/private/var/py/ReadHeart/20180830.data.sr.score.1581.json', encoding='utf-8') as f:
#     data = json.loads(f.read())


if __name__ == '__main__':
    pp = PreProcessor()
    # key = '1532069647.5589173'
    # pp.process_video(key)
    pp.process_all()
