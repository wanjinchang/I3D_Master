#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np

from models.S3D_G_Net import S3D_G_Net
from models.S3D_Net import S3D_Net
from data_providers.utils import get_data_provider_by_path

# resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(2000)
train_params = {
  'num_classes': 2,
  'batch_size': 10,
  'n_epochs': 100,
  'crop_size': (224, 224),
  'sequence_length': 10,
  'initial_learning_rate': 0.001,
  'reduce_lr_epoch_1': 30,  # epochs * 0.5
  'reduce_lr_epoch_2': 55,  # epochs * 0.75
  'validation_set': True,
  'validation_split': None,  # None or float
  'queue_size': 500,
  'normalization': 'std',  # None, divide_256, divide_255, std
}

# Separable Inception module (Sep_Inc.) parameters --> dict, each key denotes the parameters of the four branches of
# each Separable Inception block e.g. 'Inc_1': {
# 'Branch_0': [out_channels, ksize],
# 'Branch_1': [out_channels_1, ksize_1, out_channels_2, ksize_2],
# 'Branch_2': [out_channels_1, ksize_1, out_channels_2, ksize_2],
# 'Branch_3': [pool_ksize, pool_strides, out_channles, ksize]
# }
# and the kernel_size: [kernel_depth, kernel_height, kernel_width]
# the Separable Inception module (Sep_Inc.) architecture can get from the paper: https://arxiv.org/pdf/1712.04851.pdf
Sep_Inc_params = {
                  'Inc_1': {'Branch_0': [64, [1, 1, 1]],
                            'Branch_1': [96, [1, 1, 1], 128, [3, 3, 3]],
                            'Branch_2': [16, [1, 1, 1], 32, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 32, [1, 1, 1]]},
                  'Inc_2': {'Branch_0': [128, [1, 1, 1]],
                            'Branch_1': [128, [1, 1, 1], 192,  [3, 3, 3]],
                            'Branch_2': [32, [1, 1, 1], 96, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64, [1, 1, 1]]},
                  'Inc_3': {'Branch_0': [192, [1, 1, 1]],
                            'Branch_1': [96, [1, 1, 1], 208, [3, 3, 3]],
                            'Branch_2': [16, [1, 1, 1], 48, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64, [1, 1, 1]]},
                  'Inc_4': {'Branch_0': [160, [1, 1, 1]],
                            'Branch_1': [112, [1, 1, 1], 224, [3, 3, 3]],
                            'Branch_2': [24, [1, 1, 1], 64, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64, [1, 1, 1]]},
                  'Inc_5': {'Branch_0': [128, [1, 1, 1]],
                            'Branch_1': [128, [1, 1, 1], 256, [3, 3, 3]],
                            'Branch_2': [24, [1, 1, 1], 64, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64, [1, 1, 1]]},
                  'Inc_6': {'Branch_0': [112, [1, 1, 1]],
                            'Branch_1': [144, [1, 1, 1], 288, [3, 3, 3]],
                            'Branch_2': [32, [1, 1, 1], 64, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 64, [1, 1, 1]]},
                  'Inc_7': {'Branch_0': [256, [1, 1, 1]],
                            'Branch_1': [160, [1, 1, 1], 320, [3, 3, 3]],
                            'Branch_2': [32, [1, 1, 1], 128, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 128, [1, 1, 1]]},
                  'Inc_8': {'Branch_0': [256, [1, 1, 1]],
                            'Branch_1': [160, [1, 1, 1], 320, [3, 3, 3]],
                            'Branch_2': [32, [1, 1, 1], 128, [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 128, [1, 1, 1]]},
                  'Inc_9': {'Branch_0': [384, [1, 1, 1]],
                            'Branch_1': [192, [1, 1, 1], 384,  [3, 3, 3]],
                            'Branch_2': [48, [1, 1, 1], 128,  [3, 3, 3]],
                            'Branch_3': [[1, 3, 3, 3, 1], [1, 1, 1, 1, 1], 128, [1, 1, 1]]}
                 }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
    '--train', action='store_true',
    help='Train the model')
    parser.add_argument(
    '--test', action='store_true',
    help='Test model for required dataset if pretrained model exists.'
       'If provided together with `--train` flag testing will be'
       'performed right after training.')
    parser.add_argument(
    '--demo', action='store_true',
    help='the demo for required video if pretrained model exists.')
    parser.add_argument(
    '--dataset', '-ds', type=str,
    help='Path to the dataset')
    parser.add_argument(
    '--video_path1', '-ds1', type=str,
    help='Path to the video1')
    parser.add_argument(
    '--video_path2', '-ds2', type=str,
    help='Path to the video2')
    parser.add_argument(
    '--keep_prob', '-kp', type=float, default=1.0, metavar='',
    help="Keep probability for dropout.")
    parser.add_argument(
    '--gpu_id', '-gid', type=str, default='0',
    help='Specify the gpu ID to run the program')
    parser.add_argument(
    '--weight_decay', '-wd', type=float, default=1e-4, metavar='',
    help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
    '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
    help='Nesterov momentum (default: %(default)s)')

    parser.add_argument(
        '--eval_type', '-type', type=str, default='joint',
        help='use to specify the input type of the model, should be ‘rgb’, ‘flow’ , ’motempl‘, or ‘joint’')
    parser.add_argument(
    '--logs', dest='should_save_logs', action='store_true',
    help='Write tensorflow logs')
    parser.add_argument(
    '--no-logs', dest='should_save_logs', action='store_false',
    help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)

    parser.add_argument(
    '--saves', dest='should_save_model', action='store_true',
    help='Save model during training')
    parser.add_argument(
    '--no-saves', dest='should_save_model', action='store_false',
    help='Do not save model during training')
    parser.set_defaults(should_save_model=True)

    parser.add_argument(
    '--renew-logs', dest='renew_logs', action='store_true',
    help='Erase previous logs for model if exists.')
    parser.add_argument(
    '--not-renew-logs', dest='renew_logs', action='store_false',
    help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=False)

    args = parser.parse_args()
    print('args.video_path1', args.video_path1)
    print('args.video_path2', args.video_path2)

    model_params = vars(args)

    # if not args.train and not args.test or not args.dataset or not args.evaluate:
    #     print("You should train or test your network. Please check params.")
    #     parser.print_help()
    #     exit()

    # ==========================================================================
    # LIMITE THE USAGE OF THE GPU
    # =========================================================================
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ==========================================================================
    # LOG FILE SETTING
    # ==========================================================================
    # write all the log to the file without buffer
    # f = open('log.txt', 'wb', 0)
    # sys.stdout = f
    # sys.stderr = f

    # ==========================================================================
    # PARAMETERS PRINTING
    # ==========================================================================
    # some default params dataset/architecture related
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    # ==========================================================================
    # DATA PREPARATION
    # ==========================================================================
    train_params['test'] = args.test
    train_params['train'] = args.train
    if not args.train:
        train_params['validation_set'] = False
    data_provider = get_data_provider_by_path(args.dataset, train_params)

    # ==========================================================================
    # TRAINING & TESTING & EVALUATING
    # ==========================================================================
    print("Initialize the model..")
    model_params['sequence_length'] = train_params['sequence_length']
    model_params['crop_size'] = train_params['crop_size']
    model_params['Inc_params'] = Sep_Inc_params
    # print('model_params:', model_params['crop_size'])
    # print('model_params[crop_size][0]:', model_params['crop_size'][0])
    # print('model_params[crop_size][1]:', model_params['crop_size'][1])
    # create S3D_G_Net model
    model = S3D_G_Net(data_provider=data_provider, **model_params)
    # create S3D_Net model
    # model = S3D_Net(data_provider=data_provider, **model_params)
    # data_provider.data_shape
    if args.train:
        print("Data provider train videos: ", data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    # Test the trained model on test sets
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test videos: ", data_provider.test.num_examples)
        print("Testing...")
        losses = []
        accuracies = []
        for i in range(10):
            loss, accuracy = model.test(data_provider.test, batch_size=10)
            losses.append(loss)
            accuracies.append(accuracy)
        loss = np.mean(losses)
        accuracy = np.mean(accuracies)
        print("mean cross_entropy: %f, mean accuracy: %f" % (loss, accuracy))

    # evaluate the model on camera
    if args.demo:
        if not args.train and not args.test:
            model.load_model()
        print("Evaluate the trained model on video.")
        if args.eval_type == 'rgb':
            print('Evaluate the trained model on rgb input data!!!')
            model.evaluate_rgb_model(args.video_path1, args.video_path2, model_params)
        elif args.eval_type == 'flow':
            print('Evaluate the trained model on flow input data!!!')
            model.evaluate_flow_model(args.video_path1, args.video_path2, model_params)
        elif args.eval_type == 'motempl':
            print('Evaluate the trained model on motempl input data!!!')
            model.evaluate_motempl_model(args.video_path1, args.video_path2, model_params)



