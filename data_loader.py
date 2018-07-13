#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from PIL import Image
import numpy as np
import subprocess as sp
import re
import os, sys
import pickle
import cv2
import scipy.misc as sm
import shutil
from os import path

class dataLoader:
    def __init__(self,basepath,part,batchsize,fsize=128,isTrain=True):
        self.basepath = basepath
        self.part = part
        self.batchsize = batchsize
        self.fsize = fsize
        self.isTrain = isTrain
        if self.isTrain:
            self.filelistX,self.filelistY,self.filelistM = self.getallfiles()
        else:
            self.filelistX,self.filelistM = self.gettestfiles()

        self.novid = self.nofiles()

    def __len__(self):
        return len(self.filelistX)

    def nofiles(self):
        return len(self.filelistX)

    def getallfiles(self):
        fX = []
        d = self.basepath + '/' + self.part + '/X/'
        for root, _, fnames in sorted(os.walk(d)):
            fX.extend(fnames)

        fY = []
        d = self.basepath + '/' + self.part + '/Y/'
        for root, _, fnames in sorted(os.walk(d)):
            fY.extend(fnames)

        fM = []
        d = self.basepath + '/' + self.part + '/M/'
        for root, _, fnames in sorted(os.walk(d)):
            fM.extend(fnames)            

        return sorted(fX),sorted(fY),sorted(fM)

    def getYfiles(self):
        fY = []
        d = self.basepath + '/' + self.part + '/Y/'
        for root, _, fnames in sorted(os.walk(d)):
            fY.extend(fnames)

        return sorted(fY)

    def gettestfiles(self):
        fX = []
        d = self.basepath + '/' + self.part + '/X/'
        for root, _, fnames in sorted(os.walk(d)):
            fX.extend(fnames)

        fM = []
        d = self.basepath + '/' + self.part + '/M/'
        for root, _, fnames in sorted(os.walk(d)):
            fM.extend(fnames)            

        return sorted(fX),sorted(fM)       


    def getYbatch(self,idx):
        #load training data frames for stage2
        Y = []
        for i in idx:
            ok = True
            #if 1:
            try:
                Yj = []
                im = cv2.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i])
                im = cv2.resize(im, (128, 128)) 
                Yj = im[...,[2,1,0]]
                Yj = np.array(Yj, dtype='float32') / 255.
              
            except:
                print('Error clip number '+ str(i) + ' at  '+ ' OR '+ self.filelistY[i])
                ok = False
            if ok:
                Y.append(Yj)
        Y = np.asarray(Y)
        Y = Y.reshape((Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3]))
        return Y*2-1


    def getTrainbatchFrame(self,idx):
        #load training data frames for stage2
        X = []
        Y = []
        M = []
        #mapping=self.readMapping()
        #Read a batch of clips from files
        #print(idx)
        for i in idx:
            ok = True
            #if 1:
            try:
                Xj = []
                im1 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i])
                im = cv2.resize(im1, (128, 128)) 

                Xj = im[...,[2,1,0]]
                Xj = np.array(Xj, dtype='float32') / 255.


                Yj = []
                im = cv2.imread(self.basepath+'/'+self.part+'/Y/'+self.filelistY[i])
                im = cv2.resize(im, (128, 128)) 
                Yj = im[...,[2,1,0]]
                Yj = np.array(Yj, dtype='float32') / 255.

                Mj = []
                im = cv2.imread(self.basepath+'/'+self.part+'/M/'+self.filelistM[i])
                im = cv2.resize(im, (128, 128)) 
                Mj = im[...,0]
                Mj[Mj>0.5] = 255
                Mj[Mj<=0.5] = 0
                Mj = np.array(Mj, dtype='float32') / 255.                                
                
            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistY[i]+ ' OR '+ self.filelistM[i])
                ok = False
            if ok:
                X.append(Xj)
                Y.append(Yj)
                M.append(Mj)
        # make numpy and reshape
        X = np.asarray(X)
        #print(X.shape)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
        Y = np.asarray(Y)
        #print(Y.shape)
        Y = Y.reshape((Y.shape[0], Y.shape[1], Y.shape[2], Y.shape[3]))
        M = np.asarray(M)
        M = M.reshape((M.shape[0], M.shape[1], M.shape[2]))           
        #print(M.shape)
        return X*2-1, Y*2-1, M        
    
    def getTestbatch(self,idx):
        X = []
        M = []
        S = []
        for i in idx:
            ok = True
            #if 1:
            try:
                s = []
                Xj = []
                im1 = cv2.imread(self.basepath+'/'+self.part+'/X/'+self.filelistX[i])
                s = im1.shape

                im = cv2.resize(im1, (128, 128)) 
                Xj = im[...,[2,1,0]]
                Xj = np.array(Xj, dtype='float32') / 255.

                Mj = []
                im = cv2.imread(self.basepath+'/'+self.part+'/M/'+self.filelistM[i])
                im = cv2.resize(im, (128, 128)) 
                Mj = im[...,0]
                Mj[Mj>0.5] = 255
                Mj[Mj<=0.5] = 0
                Mj = np.array(Mj, dtype='float32') / 255.                                
                
            except:
                print('Error clip number '+ str(i) + ' at  '+ self.filelistX[i] + ' OR '+ self.filelistM[i])
                ok = False
            if ok:
                S.append(s)
                X.append(Xj)
                M.append(Mj)
        X = np.asarray(X)
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
        M = np.asarray(M)
        M = M.reshape((M.shape[0], M.shape[1], M.shape[2]))           
        return X*2-1, M ,S       