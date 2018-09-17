from os.path import exists, join
import json
from os import listdir
import re
from config import KEYS_FILE
import numpy as np


class ItemsCollector():
    
    def __init__(self):
        self.data_dir = './items'
        self.build_label_dict()
    
    def build_label_dict(self):
        self.label_dict = {}
        self.keys = []
        with open(KEYS_FILE, encoding='utf-8') as f:
            data = json.loads(f.read())
            for k, l in zip(data['keys'], data['labels']):
                self.label_dict[k] = l
                self.keys.append(k)
    
    def process_item(self, key):
        if not exists(join(self.data_dir, key)):
            return None
        pose_file = join(self.data_dir, key, 'poses.txt')
        if exists(pose_file):
            pose_std = json.loads(open(pose_file, encoding='utf-8').read())
            print('Pose data', pose_std)
        else:
            pose_std = [0] * 6
        
        sr_file = join(self.data_dir, key, 'sr.txt')
        if exists(sr_file):
            sr_data = json.loads(open(sr_file, 'r', encoding='utf-8').read())
        else:
            sr_data = [0.0, 0.0]
        print('SR data', sr_data)
        
        files = list(listdir(join(self.data_dir, key)))
        files = list(filter(lambda x: re.match('\d+', x), files))
        files.sort(key=lambda x: int(x))
        
        poses = []
        for file in files:
            path = join(self.data_dir, key, file, 'frame.pose.txt')
            if exists(path):
                pose = json.loads(open(path, encoding='utf-8').read())
                poses.append(pose)
        print(poses)
        
        expressions = []
        for file in files:
            path = join(self.data_dir, key, file, 'frame.emotion.l.txt')
            if exists(path):
                expression = json.loads(open(path, encoding='utf-8').read())[0]
                expressions.append(expression)
        print(expressions)
        
        expression_mean = np.mean(np.asarray(expressions), axis=0).tolist()
        print(expression_mean)
        
        label = self.label_dict[key]
        
        return {
            'pose_std': pose_std,
            'poses': poses,
            'expressions': expressions,
            'expression_mean': expression_mean,
            'label': label,
            'sr': sr_data
        }
    
    def process_all(self):
        self.items = []
        for key in self.keys:
            item = self.process_item(key)
            if item:
                self.items.append(item)
    
    def write_to_file(self):
        with open('20180917.json', 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.items))


if __name__ == '__main__':
    ic = ItemsCollector()
    # result = ic.process_item('1531992859.9004498')
    # print(result)
    ic.process_all()
    ic.write_to_file()
