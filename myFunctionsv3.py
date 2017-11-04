from tqdm import tqdm
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from random import uniform
import bcolz
import time 
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers import Input, Dense
import math
import pandas as pd
from keras.utils import np_utils as u
import utils; reload(utils)
from utils import *
import resnet50; reload(resnet50)
from resnet50 import Resnet50

import gc


def getVgg(x):
    model = vgg_ft_bn(x)
    model.compile(optimizer=Adam(1e-3),loss='binary_crossentropy', metrics=['accuracy'])
    conv_layers,fc_layers = split_at(model, Convolution2D)
    conv_model = Sequential(conv_layers)
    input_dim = conv_layers[-1].output_shape[1:]
    return (input_dim,conv_model)
    
def generateFeatures(dirname,fold, model, CHUNK_SIZE = 10000):
    
    images = load_array('03_output/folds/'+fold+'_images_0.dat')
    labels = load_array('03_output/folds/'+fold+'_labels_0.dat')

    totalImages = images.shape[0]
    batchIndex = 0
    numImages = 0
    batch_size = 64

    for i in tqdm(range(totalImages)):
        if (numImages > (CHUNK_SIZE - 1)) or (i == (totalImages-1)):

            save_array(dirname  + fold + '_labels_%d.dat'%batchIndex,
                       labels[batchIndex*CHUNK_SIZE:batchIndex*CHUNK_SIZE + numImages+1])               
            conv_feat = model.predict(images[batchIndex*CHUNK_SIZE:batchIndex*CHUNK_SIZE + numImages+1],
                                           batch_size=batch_size)                                
            save_array(dirname  + fold + '_trainX_%d.dat'%batchIndex,conv_feat[0:numImages+1]) 

            batchIndex += 1
            numImages = 0
            del(conv_feat)
            gc.collect()       

        numImages += 1
        
#TBD Early Stopping when allTag==False. Remove hardcoding of validation dataset         
def tuneMyNw(fold,folder,lr_all,e_all,p, get_model,validate=True,batch_size=64,allTags=True,base_model=None,early_stop=5,preComputed=True):

    if folder=='03_output/features':
        network='vgg'
    else:
        network='resnet'
        
    if preComputed==True:
        token = 'trainX'
    else:
        token = 'images'
    
    start = time.clock()
    featureFiles = glob(folder + '/fold_*_' + token + '_0.dat')        
    chunkCount=0
    doneOnce = False

    if validate==False:
        fold="train"

    for files in featureFiles:
        if files.find(fold) == -1:
            features = load_array(files)
            labels = load_array(files.replace(token,'labels'))
            chunkCount += 1
            numImages = features.shape[0]
            if doneOnce:
                features_all = np.vstack((features_all,features))
                labels_all = np.vstack((labels_all,labels))
            else:
                features_all = features
                labels_all = labels
                doneOnce = True
            del(features)
            del(labels)
            gc.collect()

    #featureFiles = glob(folder + '/' + fold + '_' + token + '_0.dat' )        
    featureFiles = glob('03_output/features/fold_4_trainX_0.dat' )        
    for files in featureFiles:
        features_val = load_array(files)
        labels_val = load_array(files.replace(token,'labels'))   

    if allTags==False:    
        for tag in range(1,18):

            best_loss_metric = 999

            for dropout in p:
                
                if base_model=='None':
                    bn_model = Sequential(get_model(dropout))
                if base_model=='vgg':
                    input_dim,bn_model = getVgg(17)
                    bn_model = bn_model.add(Sequential(get_model(dropout)))
                else:
                    if base_model=='resnet':
                        bn_model = getResnet()
                        bn_model = bn_model.add(Sequential(get_model(dropout)))
  
                bn_model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

                for learning_rate,epochs in zip(lr_all,e_all):
                    bn_model.optimizer.lr = learning_rate
                    for e in range(epochs):
                        bn_model.fit(features_all,labels_all[:,tag],batch_size=batch_size,nb_epoch=1,verbose=0,shuffle=True)  
                        if validate:
                            f = open('03_output/results/log.txt', 'a')       
                            mae = 0
                            loss = 0
                            avg_accuracy = 0
                            avg_loss = 0
                            totalImages = 0
                            numImages = features_val.shape[0]
                            loss_metrics = bn_model.evaluate(features_val,labels_val[:,tag],verbose=0) 
                            mae += (loss_metrics[1]*numImages)
                            loss += (loss_metrics[0]*numImages)
                            totalImages += numImages

                            if chunkCount>0: 
                                avg_mae = mae/totalImages
                                avg_loss = loss/totalImages

                                print bn_model.metrics_names,avg_loss,avg_mae, "time :", time.clock() - start
                                f.write(" : n/w : ")
                                f.write(network)
                                f.write(" : tag : ")
                                f.write(str(tag))
                                f.write(" : loss : ")
                                f.write(str(round(avg_loss,4)))
                                f.write(" : accuracy : ")
                                f.write(str(round(avg_mae,4)))
                                f.write(" : dropout : ")
                                f.write(str(dropout))
                                f.write(" : learning rate : ")
                                f.write(str(learning_rate))
                                f.write(" : time : ")
                                f.write(str(int(time.clock() - start)))
                                f.write("\n")        
                                if (avg_loss<best_loss_metric):
                                    bn_model.save_weights('03_output/results/best_'+network+'_'+fold+'_'+str(tag)+'.hdf5') 
                                    best_loss_metric = avg_loss
                            f.close() 
                        else:
                            bn_model.save_weights('03_output/results/best_'+network+'_'+fold+'_'+str(tag)+'.hdf5') 
    if allTags==True:

        best_loss_metric = 999
        tag = 'all'

        for dropout in p:
            if base_model == None:
                bn_model = Sequential(get_model(dropout))
            else:
                if base_model=='vgg':
                    input_dim,bn_model = getVgg(17)
                    bn_model.add(Sequential(get_model(dropout)))
                else:
                    if base_model=='resnet':
                        bn_model = getResnet()
                        bn_model.add(Sequential(get_model(dropout)))
            bn_model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

            for learning_rate,epochs in zip(lr_all,e_all):
                bn_model.optimizer.lr = learning_rate
                epochsIncLoss = 0
                for e in range(epochs):
                    bn_model.fit(features_all,labels_all[:,1:18],batch_size=batch_size,nb_epoch=1,verbose=0,shuffle=True) 
                    if validate:
                        f = open('03_output/results/log.txt', 'a')       
                        mae = 0
                        loss = 0
                        avg_accuracy = 0
                        avg_loss = 0
                        totalImages = 0
                        numImages = features_val.shape[0]
                        loss_metrics = bn_model.evaluate(features_val,labels_val[:,1:18],verbose=0) 
                        mae += (loss_metrics[1]*numImages)
                        loss += (loss_metrics[0]*numImages)
                        totalImages += numImages

                        if chunkCount>0: 
                            avg_mae = mae/totalImages
                            avg_loss = loss/totalImages

                            print bn_model.metrics_names,avg_loss,avg_mae, "time :", time.clock() - start
                            f.write(" : n/w : ")
                            f.write(network)
                            f.write(" : tag : ")
                            f.write(str(tag))
                            f.write(" : loss : ")
                            f.write(str(round(avg_loss,4)))
                            f.write(" : accuracy : ")
                            f.write(str(round(avg_mae,4)))
                            f.write(" : dropout : ")
                            f.write(str(dropout))
                            f.write(" : learning rate : ")
                            f.write(str(learning_rate))
                            f.write(" : time : ")
                            f.write(str(int(time.clock() - start)))
                            f.write("\n")        
                            if (avg_loss<best_loss_metric):
                                bn_model.save_weights('03_output/results/best_'+network+'_'+fold+'_'+str(tag)+'.hdf5') 
                                best_loss_metric = avg_loss
                                epochsIncLoss = 0
                            else:
                                epochsIncLoss += 1
                            f.close()
                            if epochsIncLoss>(early_stop-1):
                                epochsIncLoss = 0
                                break
                    else:
                        bn_model.save_weights('03_output/results/best_'+network+'_'+fold+'_'+str(tag)+'.hdf5') 
    return 

def loadFeatures(folder,fold,preComputed=True):
    
    #TBD Hardcoded to handle one chunk per fold
    
    if preComputed==True:
        token = 'trainX'
    else:
        token = 'images'
    
    if fold == 'test':
        featureFiles = glob(folder + '/test_*_' + token + '_0.dat')        
    else:
        if fold == 'train':
            featureFiles = glob(folder + '/fold_*_' + token + '_0.dat')        
        else:
            featureFiles = glob(folder + '/' + fold + '_' + token + '_0.dat')        
    
    doneOnce = False
    for files in featureFiles:
        features = load_array(files)
        labels = load_array(files.replace(token,'labels'))
        numImages = features.shape[0]
        if doneOnce:
            features_all = np.vstack((features_all,features))
            labels_all = np.vstack((labels_all,labels))
        else:
            features_all = features
            labels_all = labels
            doneOnce = True
            del(features)
            del(labels)
            gc.collect()
    return(features_all,labels_all)
        
def trainFolds(fold,tags,dropouts,learningSchedule,learningSchedules,epochSchedule,epochSchedules,  getmodel,network,features_all,labels_all,batch_size,allTags=True,base_model=None):

    for tag, dropout,learningScheduleID,epochScheduleID in zip(tags,dropouts,learningSchedule,epochSchedule):
        if base_model == None:
            bn_model = Sequential(get_model(dropout))
        else:
            if base_model=='vgg':
                input_dim,bn_model = getVgg(17)
            else:
                if base_model=='resnet':
                    bn_model = getResnet()
        bn_model = bn_model.add(Sequential(get_model(dropout)))
        bn_model.compile(Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
        done_once = False    
        for l1,e1 in zip(learningSchedules[learningScheduleID],epochSchedules[epochScheduleID]):
            bn_model.optimizer.lr = l1
            if allTags==True:
                bn_model.fit(features_all,labels_all[:,1:18],batch_size=batch_size,nb_epoch=e1,verbose=0,shuffle=True)
                tag == 'all'
            else:
                bn_model.fit(features_all,labels_all[:,tag],batch_size=batch_size,nb_epoch=e1,verbose=0,shuffle=True)  
        bn_model.save_weights(('03_output/results/best_'+network+ '_' + fold +'_'+str(tag)+'.hdf5') )
    
    return

def predictFolds(fold,model,network,features_val,allTags=True,base_model=None):
    dropout = 0.3 
    if base_model == None:
        bn_model = Sequential(get_model(dropout))
    else:
        if base_model=='vgg':
            input_dim,bn_model = getVgg(17)
        else:
            if base_model=='resnet':
                bn_model = getResnet()
    bn_model = bn_model.add(Sequential(get_model(dropout)))
    best_model.compile(Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    done_once = False
    if allTags==False:
        for tag in range(1,18):
            best_model.load_weights('03_output/results/best_' + network + '_' +  fold + '_' + str(tag) + '.hdf5')
            pred = best_model.predict(features_val)
            if done_once:
                pred_all =  np.hstack((pred_all,pred))
            else:
                done_once = True
                pred_all = pred
    else:
        tag = 'all'
        best_model.load_weights('03_output/results/best_' + network + '_' +  fold + '_' + str(tag) + '.hdf5')
        pred_all = best_model.predict(features_val)

    np.savetxt('03_output/results/predictions_' + network +'_' + fold + '.csv', pred_all, delimiter=",")
