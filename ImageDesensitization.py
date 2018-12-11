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

        mask[x1:x2,y1:y2,:]=255
        return mask

    def saveImage(self,savePath=''):

        cv2.imwrite(savePath,self.imageResult)



if __name__=='__main__':

    ID=ImageDesensitization()
    boundingBoxes=[{'x_top':10,'x_bottom':2000,'y_top':10,'y_bottom':2000},{'x_top':200,'x_bottom':300,'y_top':200,'y_bottom':300}]
    
    


