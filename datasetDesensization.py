#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-03 14:39:24
# @Author  : Rui Jiao (ruijiao@mail.ustc.edu.cn)
# @Link    : ${link}
# @Version : $Id$

import os
import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel
import ImageDesensitization as ID
import json
import heapq
import random


class datasetDeseneization:

    def __init__(self,datasetPath="",labelPath="",resultPath="",labelList=[]):
        self.datasetPath=datasetPath
        self.labelPath=labelPath
        self.resultPath=resultPath
        self.labelList=labelList

    def imageMosaic(self):

        imageNames=os.listdir(self.datasetPath)
        for imageName in imageNames:
            image=cv2.imread(os.path.join(self.datasetPath,imageName))
            boundingBoxes=self.getBoundingBox(imageName)
            result=ID.imageMosaic(image=image,boundingBoxes=boundingBoxes)

            cv2.imwrite(os.path.join(resultPath,imageName),result)

    def getImageShowMinError(self,number=1):

        imageNames=os.listdir(self.datasetPath)
        errors=[]
        for imageName in imageNames:
            imageSrc=cv2.imread(os.path.join(self.datasetPath,imageName))
            imageDes=cv2.imread(os.path.join(self.resultPath,imageName))
            error=np.mean(np.abs(imageSrc,imageDes))
            errors.append(error)

        minErrors=map(errors.index,heapq.nsmallest(number,errors))

        showImages=[]
        for minError in minErrors:
            showImages.append(imageNames[minError])

        return showImages

    def getImageShowRandom(self,number=1):

        imageNames=os.listdir(self.datasetPath)
        showImages=[]
        for index in random.sample(range(len(imageNames)),number):
            showImages.append(imageNames[index])
        return showImages

    def getBoundingBox(self,imageName):

        boundingBoxes=[{'x_top':10,'x_bottom':200,'y_top':10,'y_bottom':200}]
        return boundingBoxes

    def imageInpainting(self,imageHeight=512,imageWidth=680,checkpointDir=""):

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess = tf.Session(config=sess_config)
        model = InpaintCAModel()
        input_image_ph = tf.placeholder(tf.float32, shape=(1, imageHeight, imageWidth*2, 3))
        output = model.build_server_graph(input_image_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(checkpointDir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)

        imageNames=os.listdir(self.datasetPath)
        for imageName in imageNames:
            image = cv2.imread(os.path.join(self.datasetPath,imageName))
            imageShape=image.shape

            boundingBoxes=self.getBoundingBox(imageName)
            mask=np.zeros(image.shape,dtype=np.uint8)
            for boundingBox in boundingBoxes:
                x1=boundingBox['x_top']
                x2=boundingBox['x_bottom']
                y1=boundingBox['y_top']
                y2=boundingBox['y_bottom']
                mask[x1:x2,y1:y2,:]=255

            image = cv2.resize(image, (imageWidth, imageHeight))
            mask = cv2.resize(mask, (imageWidth, imageHeight))

            assert image.shape == mask.shape
            h, w, _ = image.shape
            grid = 4
            image = image[:h//grid*grid, :w//grid*grid, :]
            mask = mask[:h//grid*grid, :w//grid*grid, :] 
            #print('Shape of image: {}'.format(image.shape))
            image = np.expand_dims(image, 0)
            mask = np.expand_dims(mask, 0)
            input_image = np.concatenate([image, mask], axis=2)

            result = sess.run(output, feed_dict={input_image_ph: input_image})

            outputImage=result[0][:, :, ::-1]
            outputImage=cv2.resize(outputImage,(imageShape[1],imageShape[0]))
            cv2.imwrite(os.path.join(self.resultPath,imageName),outputImage )




if __name__=='__main__':

    datasetPath="/datapool/workspace/jiaorui/visual_desensitization/image"
    resultPath="/datapool/workspace/jiaorui/visual_desensitization/result"

    DD=datasetDeseneization(datasetPath=datasetPath,resultPath=resultPath)
    #DD.imageInpainting(imageHeight=512,imageWidth=680,checkpointDir="/datapool/workspace/jiaorui/visual_desensitization/model_logs/Places2")    
    images=DD.getImageShowMinError(number=5)
    print(images)
    images=DD.getImageShowRandom(number=5)
    print(images)