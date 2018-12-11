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

    args=parser.parse_args()

    datasetPath=args.data
    labelPath=args.label
    configPath=args.config
    resultPath=args.result
    showPath=args.show

    datasetPath="C:\\Users\\jiao\\Desktop\\visualDesensitization\\datasetJiaorui"
    labelPath="C:\\Users\\jiao\\Desktop\\visualDesensitization\\label"
    configPath='C:\\Users\\jiao\\Desktop\\visualDesensitization\\graphmask.json'
    showPath='C:\\Users\\jiao\\Desktop\\visualDesensitization\\draw'

    if(not os.path.exists(datasetPath)):
        print("the source data path does not exist")
        sys.exit(0)

    if(not os.path.exists(configPath)):
        print("missing configuration file")
        sys.exit(0)

    if(labelPath==''):
        labelPath="C:\\Users\\jiao\\Desktop\\visualDesensitization\\label"

    if(resultPath==''):
        resultPath="C:\\Users\\jiao\\Desktop\\visualDesensitization\\result"


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
        dataDesen.getImageShowRandom(number=10)
        dataDesen.drawRectangle(showPath)
    else:
        dataDesen.imageInpainting()
        dataDesen.getImageShowMinError(number=10)
        dataDesen.drawRectangle(showPath)


