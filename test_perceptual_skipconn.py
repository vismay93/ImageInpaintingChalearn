import numpy as np
import tensorflow as tf
import cv2
import tqdm
import os
import matplotlib.pyplot as plt
import sys
from network_perceptual_skipconn import Network
from data_loader import *

MODEL_PATH = './model/20latest'
DATASET_PATH = './Data'
savepath = './Output/'

IMAGE_SIZE = 128
HOLE_MIN = 24
HOLE_MAX = 48
BATCH_SIZE = 1
PRETRAIN_EPOCH = 1
#test_npy = './lfw.npy'

def test():
    with tf.device('/device:GPU:1'):
        x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
        y = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
        maskin = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE])
        is_training = tf.placeholder(tf.bool, [])

        model = Network(x, y, maskin, is_training, batch_size=BATCH_SIZE)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)
        print("model loaded")
        
        dl = dataLoader(DATASET_PATH,'test',BATCH_SIZE,IMAGE_SIZE, False)        
         
        N = dl.nofiles()
        print(N)

        writer = tf.summary.FileWriter('logs', sess.graph)
        writer.close()
        count = 0
        for i in tqdm.tqdm(range(N)):
            x_batch,m_batch,shape = dl.getTestbatch([i]) 
            files = dl.filelistX
            Y = []
            x_clip = x_batch
            m_clip = m_batch

            imitation = sess.run(model.imitation, feed_dict={x: x_clip, maskin: m_clip, is_training: False})                  
            saveOutput(savepath+'Y',files[i], shape[0],imitation[0])

def saveOutput(outpath, i,s,im):

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    im = im[...,[2,1,0]]
    im = (im+1)*255./2.
    im = cv2.resize(im, (s[1], s[0])) 
    cv2.imwrite(outpath+'/'+i,im)

if __name__ == '__main__':
    test()
