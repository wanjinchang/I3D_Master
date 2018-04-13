#!/usr/bin/env python
# encoding: utf-8
'''
@author: wanjinchang
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: wanjinchang1991@gmail.com
@software: pycharm
@file: S3D_Net.py
@time: 17-11-14 上午11:12
@desc:
'''
import os
import time
import shutil
import platform
from datetime import timedelta

import numpy as np
import cv2
import video
from common import nothing, clock, draw_str
import tensorflow as tf

action_type = {'0': 'Fighting', '1': 'Normal'}
text_color = {'Fighting': [0, 0, 255], 'Normal': [0, 255, 0]}

MHI_DURATION = 0.5
DEFAULT_THRESHOLD = 32
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05

class S3D_Net(object):
    def __init__(self, data_provider, keep_prob, dataset, weight_decay,
                 nesterov_momentum, should_save_logs, should_save_model,
                 sequence_length, crop_size, Inc_params, renew_logs=False, **kwargs):
        """
        Class to implement networks base on this paper (Inception-v1 S3D architecture).
        https://arxiv.org/pdf/1705.07750.pdf

        :param data_provider: Class, that have all required data sets.
        :param keep_prob: 'float', keep probability for dropout. If keep_prob = 1, dropout will be disables.
        :param dataset: 'str', dataset name.
        :param weight_decay: 'float', weight decay for L2 loss.
        :param nesterov_momentum: 'float', momentum for Nesterov optimizer.
        :param should_save_logs: 'bool', should logs be saved or not.
        :param should_save_model: 'bool', should model be saved or not.
        :param sequence_length: 'int', the number of input frames.
        :param crop_size: 'int', the size of input image.
        :param Inc_params: 'dict', containing the inception module parameters of I3D Net.
        :param renew_logs: 'bool', remove previous logs for current model.
        """
        self.data_provider = data_provider
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.keep_prob = keep_prob
        self.dataset_name = dataset
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.sequence_length = sequence_length
        self.crop_size = crop_size
        self.renew_logs = renew_logs
        self.Inc_params = Inc_params
        self.batches_step = 0

        self._define_inputs()
        self._build_graph()
        self._initialize_session()
        self._count_trainable_params()

    def _initialize_session(self):
        """
        Initialize session, variables, saver
        """
        # config = tf.ConfigProto()
        # # restrict model GPU memory utilization to min required
        # config.gpu_options.allow_growth = True
        # self.sess = tf.Session(config=config)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        # self.sess = tf.Session(config=tf.ConfigProto(
        #     gpu_options=gpu_options,
        #     allow_soft_placement=True,
        #     log_device_placement=False))
        self.sess = tf.Session(config=tf_config)
        self.sess.run(tf.global_variables_initializer())
        logswriter = tf.summary.FileWriter
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
        self.summary_writer = logswriter(self.logs_path, self.sess.graph)

    def _count_trainable_params(self):
        total_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_params += variable_parameters
        print('Total training parms: %.1fM' % (total_params / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
            model_path = self._model_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            if platform.python_version_tuple()[0] is '2':
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
            else:
                os.makedirs(save_path, exist_ok=True)
            model_path = os.path.join(save_path, 'model.ckpt')
            self._save_path = save_path
            self._model_path = model_path
        return save_path, model_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            logs_path = 'logs/%s' % self.model_identifier
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            if platform.python_version_tuple()[0] is '2':
                if not os.path.exists(logs_path):
                    os.makedirs(logs_path)
            else:
                os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        return "_seq_length={}_crop_size={}".format(self.sequence_length, self.crop_size)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path[1], global_step=global_step)

    def load_model(self):
        """
        load the sess from the pretrained model
        :return: start_epoch: the start step to train the model
        """
        ckpt = tf.train.get_checkpoint_state(self.save_path[0])
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            start_epoch = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            start_epoch = int(start_epoch) + 1
            print('Successfully load model from save_path: %s and epoch: %s' % (self.save_path[0], start_epoch))
            return start_epoch
        else:
            print('Training from scratch')
            return 1

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix, should_print=True):
        if should_print:
            print('mean cross entropy: %f, mean accuracy: %f' % (loss, accuracy))
        summary = tf.Summary(value=[tf.Summary.Value(tag='loss_%s' % prefix, simple_value=float(loss)),
                                    tf.Summary.Value(tag='accuracy_%s' % prefix, simple_value=float(accuracy))])
        self.summary_writer.add_summary(summary, epoch)

    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.videos = tf.placeholder(
            tf.float32,
            shape=shape,
            name='input_videos')
        self.labels = tf.placeholder(
            tf.float32,
            shape=[None, self.n_classes],
            name='labels')
        self.leaning_rate = tf.placeholder(
            tf.float32,
            shape=[],
            name='learning_rate')
        self.is_training = tf.placeholder(tf.bool, shape=[])

    def Unit3D(self, _input, _out_channels,
               ksize, padding, name, strides=[1, 1, 1, 1, 1],
               use_batch_norm=True, activation_fn=True):
        """
        Basic unit containing Conv3D + BatchNorm + non-linearity.
        :param _input: Inputs to the Unit3D component.
        :param _out_channels: the number of output filters.
        :param ksize: the filter size of shape [kernel_depth, kernel_height, kernel_width].
        :param strides: the strides of shape [1, l_st, k_zt, k_st, 1].
        :param padding: 'SAME' or 'VALID'.
        :param is_training: whether to use training mode for BatchNorm(boolean).
        :return: Outputs from the module.
        """
        net = self.conv3d(_input, _out_channels, ksize, name, strides, padding)
        if use_batch_norm:
            net = self.batch_norm(net)
        if activation_fn:
            with tf.name_scope('ReLU'):
                net = tf.nn.relu(net)
        return net

    def Sep_Conv(self, _input, _out_channels, ksize, padding, name, strides=[1, 1, 1, 1, 1],
                 use_batch_norm=True, activation_fn=True):
        """
        Basic unit containing Sep-Conv + BatchNorm + non-linearity.
        :param _input: Inputs to the Sep-Conv component.
        :param _out_channels: the number of output filters.
        :param ksize: the filter size of shape [kernel_depth, kernel_height, kernel_width].
        :param strides: the strides of shape [1, l_st, k_zt, k_st, 1].
        :param padding: 'SAME' or 'VALID'.
        :param is_training: whether to use training mode for BatchNorm(boolean).
        :return: Outputs from the module.
        """
        # spatial convolution conv_kernel_size: [1, kernel_height, kernel_width]
        with tf.name_scope('Spa_Conv'):
            net = self.conv3d(_input, _out_channels, [1, ksize[1], ksize[2]], name[0], [1, 1, strides[2], strides[3], 1],
                              padding)
        # temporal convolution conv_kernel_size: [kernel_depth, 1, 1]
        with tf.name_scope('Tem_Conv'):
            net = self.conv3d(net, _out_channels, [ksize[0], 1, 1], name[1], [1, strides[1], 1, 1, 1], padding)
        if use_batch_norm:
            net = self.batch_norm(net)
        if activation_fn:
            with tf.name_scope('ReLU'):
                net = tf.nn.relu(net)
        return net

    def Sep_Inc(self, _input, Inc_params, padding='SAME'):
        """
        Implementation the Separable Inception block(Sep_Inc.) mentioned in the paper.
        :param _input: Inputs to the Sep_Inc.
        :return: Outputs from the Sep_Inc.
        """
        with tf.variable_scope('Branch_0'):
            branch_0 = self.Unit3D(_input, Inc_params['Branch_0'][0], Inc_params['Branch_0'][1],
                                   padding, name='conv_1x1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = self.Unit3D(_input, Inc_params['Branch_1'][0], Inc_params['Branch_1'][1],
                                   padding, name='conv_1x1x1')
            branch_1 = self.Sep_Conv(branch_1, Inc_params['Branch_1'][2], Inc_params['Branch_1'][3],
                                     padding, name=['conv_1x3x3', 'conv_3x1x1'])
        with tf.variable_scope('Branch_2'):
            branch_2 = self.Unit3D(_input, Inc_params['Branch_2'][0], Inc_params['Branch_2'][1],
                                   padding, name='conv_1x1x1')
            branch_2 = self.Sep_Conv(branch_2, Inc_params['Branch_2'][2], Inc_params['Branch_2'][3],
                                     padding, name=['conv_1x3x3', 'conv_3x1x1'])
        with tf.variable_scope('Branch_3'):
            branch_3 = self.pool(_input, Inc_params['Branch_3'][0], Inc_params['Branch_3'][1], name='maxpool_3x3x3')
            branch_3 = self.Unit3D(branch_3, Inc_params['Branch_3'][2], Inc_params['Branch_3'][3],
                                   padding, name='conv_1x1x1')
        Inc_output = tf.concat([branch_0, branch_1, branch_2, branch_3], axis=4)
        return Inc_output

    def _build_graph(self):
        # end_points = {}

        # first Convolution 7x7x7
        # end_point = 'Conv3d_7x7x7'
        with tf.variable_scope('Sep_Conv_7x7x7'):
            net = self.Sep_Conv(self.videos, _out_channels=64, ksize=[7, 7, 7], padding='SAME',
                                name=['conv_1x7x7', 'conv_7x1x1'], strides=[1, 2, 2, 2, 1])
            net = self.pool(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], name='Max_pool_1x3x3')
        print('net1:', net.get_shape())  # net1: (?, 4, 56, 56, 64)

        # Convolution 3x3x3
        with tf.variable_scope('Sep_Conv_3x3x3'):
            net = self.Sep_Conv(net, _out_channels=192, ksize=[3, 3, 3], padding='SAME',
                                name=['conv_1x3x3', 'conv3x1x1'])
            net = self.pool(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1], name='Max_pool_1x3x3')
        print('net2:', net.get_shape())   # net1: (?, 4, 28, 28, 192)

        # the first Inception module (Inc_1)
        with tf.variable_scope('Inc_1'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_1'])
        print('net3:', net.get_shape())   # net3: (?, 4, 28, 28, 256)

        # the second Inception module (Inc_2)
        with tf.variable_scope('Inc_2'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_2'])
            net = self.pool(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], name='Max_pool_3x3x3')
        print('net4:', net.get_shape())   # net4: (?, 2, 14, 14, 480)

        # the third Inception module (Inc_3)
        with tf.variable_scope('Inc_3'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_3'])
        print('net5:', net.get_shape())  # net5: (?, 2, 14, 14, 512)

        # the forth Inception module (Inc_4)
        with tf.variable_scope('Inc_4'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_4'])
        print('net6:', net.get_shape())   # net6: (?, 2, 14, 14, 512)

        # the fifth Inception module (Inc_5)
        with tf.variable_scope('Inc_5'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_5'])
        print('net7:', net.get_shape())  # net7: (?, 2, 14, 14, 512)

        # the sixth Inception module (Inc_6)
        with tf.variable_scope('Inc_6'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_6'])
        print('net8:', net.get_shape())   # net8: (?, 2, 14, 14, 528)

        # the seventh Inception module (Inc_7)
        with tf.variable_scope('Inc_7'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_7'])
            net = self.pool(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], name='Max_pool_2x2x2')
        print('net9:', net.get_shape())  # net9: (?, 1, 7, 7, 832)

        # the eighth Inception module (Inc_8)
        with tf.variable_scope('Inc_8'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_8'])
        print('net10:', net.get_shape())   # net10: (?, 1, 7, 7, 832)

        # the ninth Inception module (Inc_9)
        with tf.variable_scope('Inc_9'):
            net = self.Sep_Inc(net, self.Inc_params['Inc_9'])
            net = self.pool(net, ksize=[1, 1, 7, 7, 1], strides=[1, 1, 1, 1, 1], name='Avg_pool_2x7x7',
                            _type='avg', padding='VALID')
            net = self.dropout(net, self.keep_prob)
        print('net11:', net.get_shape())  # net11: (?, 1, 1, 1, 1024)

        # the last Convolution layer 1x1x1
        with tf.variable_scope('Conv3d_1x1x1'):
            logits = self.Unit3D(net, _out_channels=self.n_classes, ksize=[1, 1, 1], use_batch_norm=False,
                                 activation_fn=False, padding='SAME', name='conv_1x1x1')
        print('net12:', logits.get_shape())  # net12: (?, 1, 1, 1, 2)

        if self.is_training is not None:
            # print('training_process!!!!!!!!!!!!!!!!!')
            logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')

        # print('here!!!!!!!!!!!!!!!')
        # print('###############:', self.is_training)
        # logits = tf.cond(self.is_training, lambda: tf.squeeze(logits, [2, 3], name='SpatialSqueeze'), lambda: logits)

        averaged_logits = tf.reduce_mean(logits, axis=1)
        # print('averaged_logits:', averaged_logits.get_shape()) averaged_logits: (?, 2)

        # features_total = output.get_shape().as_list()[-1] * output.get_shape().as_list()[-2] * \
        #                  output.get_shape().as_list()[-2]
        # output = tf.reshape(output, [-1, features_total])
        #
        # # Fully Connection Layer
        # with tf.variable_scope('fc'):
        #     W = self.weight_varibale_xavier([features_total, self.n_classes], 'weight')
        #     bias = self.bias_variable([self.n_classes], 'bias')
        #     logits = tf.add(tf.matmul(output, W), bias)
        print('logits:', logits.get_shape())
        print('averaged_logits:', averaged_logits.get_shape())

        self.logits = averaged_logits
        prediction = tf.nn.softmax(averaged_logits)
        self.prediction = prediction

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # Optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.leaning_rate, self.nesterov_momentum, use_nesterov=True)
        self.train_step = optimizer.minimize(cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def conv3d(self, _input, out_features, kernel_size, name, strides=[1, 1, 1, 1, 1], padding='SAME'):
        """
        :param _input: input to the conv3d operation.
        :param out_features: output channles for the conv3d operation.
        :param ksize: the filter size of shape [kernel_depth, kernel_height, kernel_width].
        :param strides: the strides of shape [1, l_st, k_zt, k_st, 1].
        """
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra([kernel_size[0], kernel_size[1], kernel_size[2], in_features, out_features],
                                           name=name)
        with tf.name_scope("3DConv"):
            output = tf.nn.conv3d(_input, kernel, strides, padding)
        return output

    def pool(self, _input, ksize, strides, name, _type='max', padding='SAME'):
        # if not width_k: width_k = k
        # ksize = [1, d, k, width_k, 1]
        # if not k_stride: k_stride = k
        # if not k_stride_width: k_stride_width = k_stride
        # if not d_stride: d_stride = d
        # strides = [1, d_stride, k_stride, k_stride_width, 1]
        if _type is 'max':
            output = tf.nn.max_pool3d(_input, ksize, strides, padding, name=name)
        elif _type is 'avg':
            output = tf.nn.avg_pool3d(_input, ksize, strides, padding, name=name)
        else:
            output = None
        return output

    def batch_norm(self, _input):
        with tf.name_scope('batch_normalization'):
            output = tf.contrib.layers.batch_norm(
                _input, scale=True, is_training=self.is_training,
                updates_collections=None)
        return output

    def dropout(self, _input, keep_prob):
        if keep_prob < 1:
            with tf.name_scope('dropout'):
                output = tf.cond(
                    self.is_training,
                    lambda: tf.nn.dropout(_input, keep_prob),
                    lambda: _input)
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_varibale_xavier(self, shape, name):
        return tf.get_variable(name,
                               shape=shape,
                               initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        init_learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()

        # Restore the model if we have
        start_epoch = self.load_model()

        # Start training
        for epoch in range(start_epoch, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            learning_rate = init_learning_rate

            # Update the learning rate according to the decay parameter
            if epoch >= reduce_lr_epoch_1 and epoch < reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print('Decrease learning rate, new lr = %f' % learning_rate)
            elif epoch >= reduce_lr_epoch_2:
                learning_rate = learning_rate / 100
                print('Decrease learning rate, new lr = %f' % learning_rate)

            print("Training...")
            loss, acc = self.train_one_epoch(self.data_provider.train, batch_size, learning_rate)
            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print('Validation...')
                loss, acc = self.test(self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')
            time_per_epoch = time.time() - start_time
            second_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (str(timedelta(seconds=time_per_epoch)),
                                                                str(timedelta(seconds=second_left))))

            if self.should_save_model and epoch % 10 == 0:
                self.save_model(global_step=epoch)

        total_train_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(seconds=total_train_time)))

    def train_one_epoch(self, data, batch_size, learning_rate):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            videos, labels = data.next_batch(batch_size)
            feed_dict = {
                self.videos: videos,
                self.labels: labels,
                self.leaning_rate: learning_rate,
                self.is_training: True}
            fetches = [self.train_step, self.cross_entropy, self.accuracy]
            result = self.sess.run(fetches, feed_dict=feed_dict)
            _, loss, accuracy = result
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(loss, accuracy, self.batches_step, prefix='per_batch', should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def test(self, data, batch_size):
        num_examples = data.num_examples
        total_loss = []
        total_accuracy = []
        for i in range(num_examples // batch_size):
            batch = data.next_batch(batch_size)
            feed_dict = {
                self.videos: batch[0],
                self.labels: batch[1],
                self.is_training: None
            }
            fetches = [self.cross_entropy, self.accuracy]
            loss, accuracy = self.sess.run(fetches, feed_dict=feed_dict)
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy

    def convert2bgr(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def calc_optiflow(self, last_frame, curr_frame):
        last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        # start_time = time.time()
        flow = cv2.calcOpticalFlowFarneback(last_gray, cur_gray, None, 0.5, 3, 5, 3, 5, 1.2, 0)
        # farnback_calc_time = time.time() - start_time
        # print('farnback_calc_time:', farnback_calc_time)
        bgr = self.convert2bgr(flow)
        # cv2.imshow('flow_image', bgr)
        # cv2.waitKey(0)
        return bgr

    def calc_motempl(self, last_frame, curr_frame, motion_history, timestamp):
        frame_diff = cv2.absdiff(last_frame, curr_frame)
        gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
        ret, motion_mask = cv2.threshold(gray_diff, 32, 1, cv2.THRESH_BINARY)
        # timestamp = clock()
        cv2.motempl.updateMotionHistory(motion_mask, motion_history, timestamp, MHI_DURATION)
        vis = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        return vis

    def calc_output(self, data_list, input_data):
        """
        Calculate the output using the trained model of the Net.
        :param data_list: 'list', the input data, e.g. rgb data, flow data or motempl data.
        :param input_data: ‘list’, the input of the feedict.
        :return:
            str action: the type of action.
            float prob: the prob of the action.
            result: list, the output of the model.
        """
        input_data.append(data_list)
        data = np.array(input_data).astype(np.float32)
        start_time = time.time()
        feed_dict = {self.videos: data, self.is_training: None}
        fetches = [self.prediction]
        result = self.sess.run(fetches, feed_dict=feed_dict)
        duration = time.time() - start_time
        print('result_video: ', result)
        print('forward_duration: %s' % duration)
        index = np.argmax(result)
        print(index)
        type_id = str(index)
        # print('type_id:', type_id)
        action = action_type[type_id]
        print('type: ', action)
        prob = result[0][0][index]
        return action, prob, result

    def evaluate_flow_model(self, video_path1, video_path2, model_params):
        camera = cv2.VideoCapture(video_path1)
        camera2 = cv2.VideoCapture(video_path2)
        assert camera.isOpened() and camera2.isOpened(), 'Can not capture source!'
        flow_data = []
        img_data = []
        input_data = []
        action = 'Normal'
        prob = 1.0

        flow_data2 = []
        img_data2 = []
        input_data2 = []
        action2 = 'Normal'
        prob2 = 1.0
        # i = 0
        while camera.isOpened() and camera2.isOpened():
            try:
                _, frame = camera.read()
                _, frame2 = camera2.read()
                temp_frame = cv2.resize(frame, (model_params['crop_size'][0], model_params['crop_size'][1]),
                                        interpolation=cv2.INTER_CUBIC)
                temp_frame2 = cv2.resize(frame2, (model_params['crop_size'][0], model_params['crop_size'][1]),
                                         interpolation=cv2.INTER_CUBIC)
                img_data.append(temp_frame)
                img_data2.append(temp_frame2)

                # Calculate the optical flow between two frames of camera1
                if len(img_data) == 2:
                    flow_img = self.calc_optiflow(img_data[0], img_data[1])
                    # flow_img = flow_img * 1.0 / 127.5
                    flow_img = np.array(flow_img)
                    flow_data.append(flow_img)
                    img_data = []

                # Calculate the optical flow between two frames of camera2
                if len(img_data2) == 2:
                    flow_img2 = self.calc_optiflow(img_data2[0], img_data2[1])
                    # flow_img2 = flow_img2 * 1.0 / 127.5
                    flow_img2 = np.array(flow_img2)
                    flow_data2.append(flow_img2)
                    img_data2 = []

                # camera1
                if len(flow_data) == model_params['sequence_length']:
                    action, prob, _ = self.calc_output(flow_data, input_data)
                    flow_data = []
                    input_data = []

                # camera2
                if len(flow_data2) == model_params['sequence_length']:
                    action2, prob2, _ = self.calc_output(flow_data2, input_data2)
                    flow_data2 = []
                    input_data2 = []

                cv2.putText(frame, action, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action], 3)
                cv2.putText(frame, str(prob), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action], 3)
                cv2.putText(frame2, action2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action2], 3)
                cv2.putText(frame2, str(prob2), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action2], 3)
                cv2.imshow('camera1', frame)
                cv2.imshow('camera2', frame2)
                choice = cv2.waitKey(10)
                choice = cv2.waitKey(10)

            except Exception as e:
                print(e)
                camera = cv2.VideoCapture(video_path1)
                camera2 = cv2.VideoCapture(video_path2)

    def evaluate_rgb_model(self, video_path1, video_path2, model_params):
        camera = cv2.VideoCapture(video_path1)
        camera2 = cv2.VideoCapture(video_path2)
        assert camera.isOpened() and camera2.isOpened(), 'Can not capture source!'
        img_data = []
        input_data = []
        action = 'Normal'
        prob = 1.0

        img_data2 = []
        input_data2 = []
        action2 = 'Normal'
        prob2 = 1.0
        while camera.isOpened() and camera2.isOpened():
            try:
                _, frame = camera.read()
                _, frame2 = camera2.read()
                temp_frame = cv2.resize(frame, (model_params['crop_size'][0], model_params['crop_size'][1]),
                                        interpolation=cv2.INTER_CUBIC)
                temp_frame2 = cv2.resize(frame2, (model_params['crop_size'][0], model_params['crop_size'][1]),
                                         interpolation=cv2.INTER_CUBIC)
                # temp_frame = self.normalize_image(temp_frame, 'std')
                # temp_frame2 = self.normalize_image(temp_frame2, 'std')
                # temp_frame = temp_frame / 127.5 - 1
                # temp_frame2 = temp_frame2 / 127.5 - 1
                # temp_frame2 = self.normalize_image(temp_frame2, 'std')
                temp_frame = temp_frame * 1.0 / 127.5
                temp_frame2 = temp_frame2 * 1.0 / 127.5
                temp_frame = np.array(temp_frame)
                temp_frame2 = np.array(temp_frame2)

                img_data.append(temp_frame)
                img_data2.append(temp_frame2)

                # camera1
                if len(img_data) == model_params['sequence_length']:
                    action, prob, _ = self.calc_output(img_data, input_data)
                    img_data = []
                    input_data = []

                # camera2
                if len(img_data2) == model_params['sequence_length']:
                    print('camera2!!!!!!')
                    action2, prob2, _ = self.calc_output(img_data2, input_data2)
                    img_data2 = []
                    input_data2 = []

                cv2.putText(frame, action, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[action], 3)
                cv2.putText(frame, str(prob), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action], 3)
                cv2.putText(frame2, action2, (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, text_color[action2], 3)
                cv2.putText(frame2, str(prob2), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action2], 3)
                cv2.imshow('camera1', frame)
                cv2.imshow('camera2', frame2)
                choice = cv2.waitKey(10)
                choice = cv2.waitKey(10)

            except Exception as e:
                print(e)
                camera = cv2.VideoCapture(video_path1)
                camera2 = cv2.VideoCapture(video_path2)

    def evaluate_motempl_model(self, video_path1, video_path2, model_params):
        camera = cv2.VideoCapture(video_path1)
        camera2 = cv2.VideoCapture(video_path2)
        assert camera.isOpened() and camera2.isOpened(), 'Can not capture source!'
        flow_data = []
        img_data = []
        input_data = []
        # model_params['crop_size']=224
        motion_history = np.zeros((model_params['crop_size'][0], model_params['crop_size'][1]), np.float32)
        action = 'Normal'
        prob = 1.0

        flow_data2 = []
        img_data2 = []
        input_data2 = []
        action2 = 'Normal'
        motion_history2 = np.zeros((model_params['crop_size'][0], model_params['crop_size'][1]), np.float32)
        prob2 = 1.0
        while camera.isOpened() and camera2.isOpened():
            try:
                _, frame = camera.read()
                _, frame2 = camera2.read()
                temp_frame = cv2.resize(frame, (model_params['crop_size'][0], model_params['crop_size'][1]),
                                        interpolation=cv2.INTER_CUBIC)
                temp_frame2 = cv2.resize(frame2, (model_params['crop_size'][0], model_params['crop_size'][1]),
                                         interpolation=cv2.INTER_CUBIC)
                img_data.append(temp_frame)
                img_data2.append(temp_frame2)

                # Calculate the motempl flow between two frames of camera1
                if len(img_data) == 3:
                    timestamp = clock()
                    flow_img = self.calc_motempl(img_data[0], img_data[2], motion_history, timestamp)
                    flow_img = flow_img * 1.0 / 127.5
                    cv2.imshow('mote1', flow_img)
                    flow_img = np.array(flow_img)
                    flow_data.append(flow_img)
                    img_data = []

                # Calculate the motempl flow between two frames of camera2
                if len(img_data2) == 3:
                    timestamp2 = clock()
                    flow_img2 = self.calc_motempl(img_data2[0], img_data2[2], motion_history2, timestamp2)
                    flow_img2 = flow_img2 * 1.0 / 127.5
                    cv2.imshow('mote2', flow_img2)
                    flow_img2 = np.array(flow_img2)
                    flow_data2.append(flow_img2)
                    img_data2 = []

                # camera1
                if len(flow_data) == model_params['sequence_length']:
                    action, prob, _ = self.calc_output(flow_data, input_data)
                    flow_data = []
                    input_data = []

                # camera2
                if len(flow_data2) == model_params['sequence_length']:
                    action2, prob2, _ = self.calc_output(flow_data2, input_data2)
                    flow_data2 = []
                    input_data2 = []

                cv2.putText(frame, action, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action], 3)
                cv2.putText(frame, str(prob), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action], 3)
                cv2.putText(frame2, action2, (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action2], 3)
                cv2.putText(frame2, str(prob2), (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                            1, text_color[action2], 3)
                cv2.imshow('camera1', frame)
                cv2.imshow('camera2', frame2)
                choice = cv2.waitKey(10)
                choice = cv2.waitKey(10)

            except Exception as e:
                print(e)
                camera = cv2.VideoCapture(video_path1)
                camera2 = cv2.VideoCapture(video_path2)

