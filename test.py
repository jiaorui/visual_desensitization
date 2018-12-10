#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-12-10 18:33:45
# @Author  : Rui Jiao (ruijiao@mail.ustc.edu.cn)
# @Link    : ${link}
# @Version : $Id$

import pickle



data=pickle.load(open("/datapool/workspace/yuanmu/demo_labelset/Stanford40/run_facerecognition_imgpath2res.pkl","rb"))
print(data)
