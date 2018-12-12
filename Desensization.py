#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-10 15:22:53
# @Author  : Rui Jiao (ruijiao@mail.ustc.edu.cn)
# @Link    : ${link}
# @Version : $Id$

import os
import sys
import datasetDesensization as DD
import ImageDesensitization as ID
import numpy as np
import argparse
import json
import neuralgym as ng


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='', type=str,
                    help='The file path of dataset to be desensitized.')
parser.add_argument('--label', default='', type=str,
                    help='The file path of dataset label')
parser.add_argument('--config', default='', type=str,
                    help='config file: what and how about desensization')
parser.add_argument('--result', default='', type=str,
                    help='The file path of result')
parser.add_argument('--show',default='',type=str,
                    help='The path of show image')

if __name__=="__main__":

    ng.get_gpus(1)
    args=parser.parse_args()

    datasetPath=args.data
    labelPath=args.label
    configPath=args.config
    resultPath=args.result
    showPath=args.show

    path="."
    datasetPath=os.path.join(path,"datasetJiaorui")
    labelPath=os.path.join(path,"label")
    configPath=os.path.join(path,"graphmask.json")
    showPath=os.path.join(path,"draw")
    resultPath=os.path.join(path,"result")

    if(not os.path.exists(datasetPath)):
        print("the source data path does not exist")
        sys.exit(0)

    if(not os.path.exists(configPath)):
        print("missing configuration file")
        sys.exit(0)

    if(labelPath==''):
        print("please input the label path: --label + label path")
        sys.exit(0)

    if(resultPath==''):
        print("please input the result path: --result + result path")
        sys.exit(0)

    if(showPath==''):
        print('please input the path where the images of showing: --show + show path')
        sys.exit(0)



    with open(configPath) as f_load:
        config=json.load(f_load)

    method=config["method"]
    content=config["content"]

    labelList=[]
    for key,value in content.items():
        if(value==1):
            labelList.append(key.replace("_"," "))

    dataDesen=DD.datasetDeseneization(datasetPath=datasetPath,labelPath=labelPath,
        resultPath=resultPath,labelList=labelList)

    if(method==0):
        dataDesen.imageMosaic()
        dataDesen.getImageShowRandom(number=100)
        dataDesen.drawRectangle(showPath)
    else:
        #the image height and width is associted to the dataset , now we use the [places] dataset
        checkpointDir=os.path.join(path,'model_logs/Places2')
        dataDesen.imageInpainting(imageHeight=512,imageWidth=680,checkpointDir=checkpointDir)
        dataDesen.getImageShowMinError(number=100)
        dataDesen.drawRectangle(showPath)


