"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json

import tensorflow as tf
import numpy as np
import foolbox
import matplotlib.pyplot as plt
import png



import cifar10_input

parser = argparse.ArgumentParser(description='TF CIFAR PGD')
parser.add_argument('--model-ckpt', default='/data/hzzheng/Code.baseline-atta.cifar10_challenge.10.21/data-model/m.10.model/checkpoint-44000',
                    help='Log path.')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--atta-loop', type=int, default=10,
                    help='ATTA attack measurement loop.')
parser.add_argument('--model-name', default='m.3.model',
                    help='model name')
parser.add_argument('--model-dir', default='./models/data-model/',
                    help='The dir of the saved model')
parser.add_argument('--ckpt-step', type=int, default=4000,
                    help='checkpoint step')
parser.add_argument('--ckpt', type=int, default=0,
                    help='checkpoint')
parser.add_argument('--ckpt-start', type=int, default=0,
                    help='checkpoint')
parser.add_argument('--ckpt-end', type=int, default=69000,
                    help='checkpoint')
parser.add_argument('--batch-size', type=int, default=128,
                    help='checkpoint')
parser.add_argument('--data-size', type=int, default=10000,
                    help='checkpoint')
args = parser.parse_args()

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# log_file = open(args.log_path, 'w')

if __name__ == '__main__':
    import json

    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)


    model = Model('eval')
    logits = model.pre_softmax
    input_ph = model.x_input
    labels_ph = model.y_input
    loss = model.mean_xent
    saver = tf.train.Saver()

    # Setup the parameters
    epsilon = 0.031  # Maximum perturbation
    batch_size = 128

    model_ckpt = args.model_ckpt

    # (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    # x_test = x_test[0:20, :]
    # y_test = y_test[0:20]

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)
    x_batch = cifar.eval_data.xs[0:50, :]
    y_batch = cifar.eval_data.ys[0:50]
    # print(x_test.shape)
    # print(min_pixel_value)
    # print(max_pixel_value)


    with tf.Session() as sess:
        saver.restore(sess, model_ckpt)
        fmodel = foolbox.models.TensorFlowModel(input_ph, logits, (0, 255))
        # print(np.argmax(fmodel.forward_one(x_batch[0])))
        print(x_batch[0].shape)
        x = np.transpose(x_batch[0], (1, 2, 0)).copy()
        png.from_array(x, 'L').save("nat.png")
        
        # plt.imshow(x_batch[0])
        print(np.mean(fmodel.forward(x_batch).argmax(axis=-1) == y_batch))
        attack = foolbox.attacks.PGD(fmodel, distance=foolbox.distances.Linf)
        # attack.as_generator(fmodel, epsilon=0.031, stepsize=0.0031, iterations=20)
        x_batch_adv = attack(x_batch, y_batch, epsilon=0.031, stepsize=0.0031, iterations=1, unpack=False)
        # x_batch_adv = attack(x_batch, y_batch, epsilon=0.031, stepsize=0.0031, iterations=20)
        # x_batch_adv = attack(x_batch, y_batch, max_epsilon=0.031, unpack=False)        
        # png.from_array(x_batch[0]).save("nat.png")
        # png.from_array(x_batch_adv[0]).save("adv.png")

        # plt.imshow(x_batch[0])
        # plt.imshow(x_batch_adv[0])

        nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
        # adv_dict = {model.x_input: x_batch_adv,
        #                 model.y_input: y_batch}

        nat_corr = sess.run(model.accuracy, feed_dict=nat_dict)
        # adv_corr = sess.run(model.accuracy, feed_dict=adv_dict)
        distances = np.asarray([a.distance.value for a in x_batch_adv])
        adv_acc = np.mean(np.isinf(distances))
        

        print("batch nat acc: {}, adv acc: {}".format(nat_corr, adv_acc))
        print("{} of {} attacks failed".format(sum(adv.distance.value == np.inf for adv in x_batch_adv), len(x_batch_adv)))


        
        # print(np.argmax(fmodel.forward_one(adv)))
        


        # predictions = classifier.predict(x_test)
        # print(x_test[0])
        # # print(predictions)
        
        # print(np.argmax(predictions, axis=1))
        # accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
        # print('Accuracy on benign test examples: {}%'.format(accuracy * 100))

