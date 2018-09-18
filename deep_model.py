import json
import tensorflow as tf
from config import IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, IMAGE_FOR_EXPRESSION_HEIGHT, \
    IMAGE_FOR_EXPRESSION_WIDTH, LANDMARK_TENSORS_PATH, EXPRESSION_TENSORS_PATH, MAP_JSON_FILE, VIDEOS_PATH
from utils import load_tensors
import logging
import numpy as np
from os import listdir

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)


class Model():
    
    def __init__(self):
        self.logger = logger
        self.learning_rate = 0.001
        
        self.build_graph()
        # self.build_model()
    
    def _expression_nn(self, inputs):
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
            name='expression_dense1')
        
        dense2 = tf.layers.dense(
            inputs=dense1,
            units=192,
            name='expression_dense2')
        
        logits = tf.layers.dense(
            inputs=dense2,
            units=7,
            name='expression_logits')
        
        return logits
    
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
        
        # logits = tf.reshape(logits, shape=[-1, 68, 2])
        return logits
    
    def build_cell(self, name):
        cell = tf.nn.rnn_cell.GRUCell(num_units=self.hidden_units, name=name)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_rate)
        return cell
    
    def build_graph(self):
        self.batch_size = 12
        self.max_frame = 100
        self.hidden_units = 200
        self.keep_rate = 0.7
        # expression graph
        self.expression_inputs = tf.placeholder(
            shape=[self.batch_size, self.max_frame, IMAGE_FOR_EXPRESSION_WIDTH, IMAGE_FOR_EXPRESSION_HEIGHT],
            dtype=tf.float32)
        self.expression_inputs_legnth = tf.placeholder(
            shape=[self.batch_size], dtype=tf.float32)
        self.expression_logits = self._expression_nn(self.expression_inputs)
        print('Expression logits', self.expression_logits)
        self.expression_logits_reshape = tf.reshape(self.expression_logits, shape=[-1, self.max_frame, 7])
        print('expression_logits_reshape', self.expression_logits_reshape)
        # self.expression_probs = tf.nn.softmax(self.expression_logits)
        self.expression_dif = self.expression_logits_reshape[:, 1:] - self.expression_logits_reshape[:, :-1]
        print('Expression dif', self.expression_dif)
        
        self.expression_dif_reshape = tf.reshape(self.expression_dif, shape=[-1, self.max_frame - 1, 7])
        print('expression_dif_reshape', self.expression_dif_reshape)
        
        # landmark graph
        self.landmark_inputs = tf.placeholder(
            shape=[self.batch_size, self.max_frame, IMAGE_FOR_LANDMARK_WIDTH, IMAGE_FOR_LANDMARK_HEIGHT, 3],
            dtype=tf.float32)
        self.landmark_inputs_length = tf.placeholder(
            shape=[self.batch_size], dtype=tf.float32)
        
        self.landmark_logits = self._landmark_nn(self.landmark_inputs)
        print('Landmark logits', self.landmark_logits)
        
        self.landmark_logits_reshape = tf.reshape(self.landmark_logits, shape=[-1, self.max_frame, 136])
        print('landmark_logits_reshape', self.landmark_logits_reshape)
        
        self.landmark_dif = self.landmark_logits_reshape[:, 1:] - self.landmark_logits_reshape[:, :-1]
        print('Landmark dif', self.landmark_dif)
        
        self.landmark_dif_shape = tf.reshape(self.landmark_dif, shape=[-1, self.max_frame - 1, 136])
        print('landmark_dif_shape', self.landmark_dif_shape)
        
        cell = self.build_cell(name='expression_cell')
        print(cell)
        self.expression_outputs, self.expression_state = tf.nn.dynamic_rnn(cell=cell,
                                                                           inputs=self.expression_dif_reshape,
                                                                           sequence_length=self.expression_inputs_legnth,
                                                                           dtype=tf.float32)
        cell = self.build_cell(name='landmark_cell')
        print(cell)
        self.landmark_outputs, self.landmark_state = tf.nn.dynamic_rnn(cell=cell,
                                                                       inputs=self.landmark_dif_shape,
                                                                       sequence_length=self.landmark_inputs_length,
                                                                       dtype=tf.float32)
        
        print('Expression outputs', self.expression_state)
        
        print('landmark_state', self.landmark_state)
        
        # sr_score_pairs_inputs: [batch_size, 2]
        self.sr_pairs_inputs = tf.placeholder(shape=[None, 2], dtype=tf.float32)
        self.logger.debug('sr_pairs_inputs %s', self.sr_pairs_inputs)
        
        self.labels_inputs = tf.placeholder(shape=[None], dtype=tf.int32)
        self.logger.debug('labels inputs %s', self.labels_inputs)
        
        self.sr_pairs_state = tf.layers.dense(inputs=self.sr_pairs_inputs, units=self.hidden_units,
                                              activation=tf.nn.relu,
                                              name='sr_pairs_dense')
    
    def build_model(self):
        landmark_tensors = load_tensors(LANDMARK_TENSORS_PATH)
        expression_tensors = load_tensors(EXPRESSION_TENSORS_PATH)
        tensors = {**landmark_tensors, **expression_tensors}
        print(tensors)
        self.sess = tf.Session()
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(v)
            v.load(tensors[v.name], self.sess)
        
        self.concated_inputs = tf.concat([self.expression_state, self.landmark_state, self.sr_pairs_state], axis=-1)
        
        self.logger.debug('concated_inputs %s', self.concated_inputs)
        
        self.inputs_dense = tf.layers.dense(self.concated_inputs, units=self.hidden_units, name='first_dense',
                                            activation=tf.nn.relu, )
        
        self.logger.debug('inputs_dense %s', self.inputs_dense)
        
        self.logists = tf.layers.dense(self.inputs_dense, units=2, name='second_dense')
        self.logger.debug('logists %s', self.logists)
        
        self.logists_softmax = tf.nn.softmax(self.logists)
        self.logger.debug('logists_softmax %s', self.logists_softmax)
        
        self.predict = tf.argmax(self.logists_softmax, axis=-1)
        self.logger.debug('predict %s', self.predict)
        
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(self.predict, tf.cast(self.labels_inputs, tf.int64)), tf.float32))
        self.logger.debug('accuracy %s', self.accuracy)
        
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logists, labels=self.labels_inputs))
        self.logger.debug('loss %s', self.loss)
        
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        self.merged_summary = tf.summary.merge_all()
    
    def build_optimizer(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def build_matrix(self, batch):
        lengths = []
        pad = [0] * len(batch[0][0])
        max_length = 0
        for item in batch:
            lengths.append(len(item))
            if max_length < len(item):
                max_length = len(item)
        for item in batch:
            d = max_length - len(item)
            for _ in range(d):
                item.append(pad)
        return np.asarray(batch), np.asarray(lengths)
    
    def prepare_data(self):
        
        map_json = json.loads(open(MAP_JSON_FILE, encoding='utf-8').read())
        self.total_size = len(map_json)
        
        # path = './data/attitudes/videos'
        files = list(listdir(VIDEOS_PATH))
        self.x_data_videos = files
        print(self.x_data_videos)
        self.x_data_srs = list(map(lambda x: map_json[x.replace('.mp4', '')]['sr'], self.x_data_videos))
        print(self.x_data_srs)
        
        self.y_data = list(map(lambda x: map_json[x.replace('.mp4', '')]['label'], self.x_data_videos))
        print(self.y_data)
        
        print(self.x_data_srs, self.y_data)
    
    def prepare_batch(self, videos, srs, labels):
        for video, sr, label in zip(videos, srs, labels):
            print(video, sr, label)
            video_path = join(VIDEOS_PATH, video)
    
    def prepare_train_data(self):
        steps = int(self.total_size / self.batch_size) + 1
        for step in range(steps):
            start_index = step * self.batch_size
            end_index = (step + 1) * self.batch_size
            train_batch = self.prepare_batch(videos=self.x_data_videos[start_index:end_index],
                                             srs=self.x_data_srs[start_index:end_index],
                                             labels=self.y_data[start_index:end_index]
                                             )
            print(train_batch)
            
            
            
    
    # def split(self):
    
    # for file in files:
    
    # for file in files:
    #     print(file)
    #
    # video_path = self.get_video_path()
    # video = cv2.VideoCapture(video_path)
    # video = cv2.VideoCapture(VIDEO)
    
    # flag = True
    # count = 1
    # poses = []
    # while flag:
    #     flag, frame = video.read()
    #     print(frame)
    #     if not flag:
    #         break
    #     try:
    #         # cv2.imwrite(join('video_ouput1'))
    #         self.er.process(deepcopy(frame), join(self.output_path, key, str(count)), 'frame')
    #         pose = self.pr.process(deepcopy(frame), join(self.output_path, key, str(count)), 'frame')
    #         poses.append(pose)
    #     except:
    #         pass
    #     count += 1


if __name__ == '__main__':
    m = Model()
    m.prepare_data()
