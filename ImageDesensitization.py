#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-02 16:35:22
# @Author  : Rui Jiao (ruijiao@mail.ustc.edu.cn)
# @Link    : ${link}
# @Version : $Id$
import os
import numpy as np
import cv2
import heapq



class ImageDesensitization:

    def imageInpaintingOpenCV(self,image,imageMask,inpaintRadius=3,flags=cv2.INPAINT_TELEA):
        
        self.imageResult=cv2.inpaint(image,imageMask,inpaintRadius,flags)

    def mosaic(self,selectedImage,nsize):

        rows,cols,_ = selectedImage.shape
        dist = selectedImage.copy()
   
        for y in range(0,rows,nsize):
            for x in range(0,cols,nsize):
                dist[y:y+nsize,x:x+nsize] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        return dist

    def imageMosaic(self,image,boundingBoxes,nsize=9):

        srcCopy=image.copy()
        for boundingBox in boundingBoxes:
            x1=boundingBox['x_top']
            x2=boundingBox['x_bottom']
            y1=boundingBox['y_top']
            y2=boundingBox['y_bottom']

            selectedImage=srcCopy[y1:y2,x1:x2]
            mosaicArea=self.mosaic(selectedImage,nsize)
            srcCopy[y1:y2,x1:x2]=cv2.addWeighted(mosaicArea,0.65,selectedImage,0.35,0)
        return srcCopy

    def imageMaskGeneration(self,imageShape,boundingBox):

        mask=np.zeros(imageShape,dtype=np.uint8)

        x1=boundingBox['x_top']
        x2=boundingBox['x_bottom']
        y1=boundingBox['y_top']
        y2=boundingBox['y_bottom']

        mask[y1:y2,x1:x2,:]=255
        return mask

    def imageBlur(self,image,boundingBoxes):

        for boundingBox in boundingBoxes:
            x1=boundingBox['x_top']
            x2=boundingBox['x_bottom']
            y1=boundingBox['y_top']
            y2=boundingBox['y_bottom']

            imageBlurSrc=image[y1:y2,x1:x2,:]
            imageBlurDes=cv2.GaussianBlur(imageBlurSrc,ksize=(25,25),sigmaX=0)
            image[y1:y2,x1:x2,:]=imageBlurDes
        return image


    def saveImage(self,savePath=''):

        cv2.imwrite(savePath,self.imageResult)



if __name__=='__main__':

    ID=ImageDesensitization()
    boundingBoxes=[{'x_top':10,'x_bottom':200,'y_top':10,'y_bottom':200},{'x_top':200,'x_bottom':300,'y_top':200,'y_bottom':300}]
    imagePath="C:\\Users\\jiao\\Desktop\\visualDesensitization\\datasetFace\\applauding_002.jpg"

    image=cv2.imread(imagePath)
    des=ID.imageBlur(image,boundingBoxes)
    #des=cv2.GaussianBlur(image,ksize=(25,25),sigmaX=0)
    cv2.imshow("retult",des)
    cv2.waitKey()
    cv2.destroyAllWindows()


