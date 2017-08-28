import os
import random
from scipy.misc import imread,imresize,imshow
from scipy.ndimage import shift,rotate
import numpy as np

# import matplotlib.pyplot as plt
def loadTransform(imgPath, s=(0,0), ang=0., size=(20.20),testSet=False):
    # shift(s) + SmlRotate + resize( (20,20) , (28,28) )
    # optional:BigRotate + horizontal flip(the flip of the character is not the same as the original character)
    img = imread(imgPath)/255
    if not testSet:
        # shift
        imgS=shift(img,s,cval=1)
        # rotate
        imgR=rotate(imgS,ang,mode='constant',cval=1)
        # imgR = np.maximum(np.minimum(rotate(imgS, ang, cval=1),1.),0.)
        img=imgR
    # # resize
    img=imresize(img,size)
    # plt.figure()
    # plt.imshow(img)
    # plt.figure()
    # plt.subplot(221)
    # plt.imshow(img)
    # plt.subplot(222)
    # plt.imshow(imgS)
    # plt.subplot(223)
    # plt.imshow(imgR)
    # plt.subplot(224)
    # plt.imshow(imgR1)
    return img


def getShuffleImg(clsFolders, nbSmpsPerCls):
    """
    :param clsFolders: clsFolders should be absolute path
    :param nbSmpsPerCls:
    :return:
    """
    imgs=[]
    for i,folder in enumerate(clsFolders):
        imgsAFolder=random.sample(os.listdir(folder),nbSmpsPerCls)
        imgsAFolder=[(i,os.path.join(folder,imgPath)) for imgPath in imgsAFolder]
        imgs.extend(imgsAFolder)
    random.shuffle(imgs)
    return imgs

