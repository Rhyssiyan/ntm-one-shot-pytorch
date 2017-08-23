import os
import random

import numpy as np

from utils import loadTransform, getShuffleImg


class OmniglotDataset(object):

    def __init__(self,folderName,
                 batchSize=16,
                 nbClsPerEpi=5,
                 nbSmpsPerCls=10,
                 size=(20,20),
                 maxShift=10,
                 maxIter=None,
                 maxSmlRot=15,# should the unit of rotation be radian
                 projAbsPath="/home/lenovo/dev/projects/pytorch/ntm-one-shot"
                 ):
        """

        :param folderName: evaluation or train
        :param batchSize:
        :param nbClsPerEpi:
        :param nbSmpsPerCls:
        :param size:
        :param maxShift:
        :param maxIter:
        :param maxSmlRot:
        :param projAbsPath:
        """
        super(OmniglotDataset,self).__init__()
        #config
        self.imgSize    = size
        self.pixelNum   = self.imgSize[0] * self.imgSize[1]
        self.curIter    = 0

        self.batchSize   =batchSize
        self.nbClsPerEpi =nbClsPerEpi
        self.nbSmpsPerCls=nbSmpsPerCls
        self.maxShift    =maxShift
        self.maxSmlRot   =maxSmlRot
        self.projAbsPath =projAbsPath
        self.seqLen      =self.nbSmpsPerCls * self.nbClsPerEpi
        self.maxIter     =maxIter
        self.dataAbsPath = os.path.join(projAbsPath,"data")
        self.folder      = os.path.join(self.dataAbsPath,folderName)
        self.fileList    = [os.path.join(self.folder,subfolder,subsubfolder) for subfolder in os.listdir(self.folder) \
                            for subsubfolder in os.listdir(os.path.join(self.folder,subfolder))]
        random.shuffle(self.fileList)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.maxIter is None or self.curIter<self.maxIter:
            self.curIter += 1
            epiData = self.getNextEpisode()
            return (self.curIter,epiData)
        raise StopIteration()

    @profile
    def getNextEpisode(self):
        """
        One-hot encoding
        :return:
        """
        # randomly sample classes of episode
        # randomly sample from classes compose batch_size * seq_len * img_vec
        # For generality, the file path in fileList would be relative path

        clsFolders=random.sample(self.fileList,self.nbClsPerEpi) #determine which classes to sample

        inputx = np.zeros((self.batchSize, self.seqLen,self.pixelNum+self.nbClsPerEpi), dtype=float)
        labels = np.zeros((self.batchSize, self.seqLen), dtype=int)
        for i in range(self.batchSize):
            # bigRotate = random.randint(0, 3) * 90
            bigRotate=0
            labelsAndImgs = getShuffleImg(clsFolders, self.nbSmpsPerCls)
            labelsPerBatch, imgs  = zip(*labelsAndImgs)
            labelsOneHot=np.zeros((self.seqLen,self.nbClsPerEpi), dtype=float)
            labelsOneHot[range(1,self.seqLen), labelsPerBatch[:-1]]=1
            shifts= np.random.randint(-self.maxShift,self.maxShift,size=(self.seqLen,2))
            rots  = np.add(np.random.uniform(-self.maxSmlRot,self.maxSmlRot,size=self.seqLen),bigRotate)

            # inputx[i] = np.asarray([np.concatenate((loadTransform(img,shift,rot+bigRotate,self.imgSize).flatten(), labelOneHot)) \
            #                         for img,shift,rot,labelOneHot in zip(imgs,shifts,rots,labelsOneHot)])
            # with Pool(processes=2) as p:
                # p.starmap(self.cmbAndTransImg,zip(imgs,shifts,rots,labelsOneHot))
            inputx[i] = np.asarray([np.concatenate((loadTransform(img,shift,rot+bigRotate,self.imgSize).flatten(), labelOneHot)) \
                                     for img,shift,rot,labelOneHot in zip(imgs,shifts,rots,labelsOneHot)])
            labels[i]    = np.asarray(labelsPerBatch)
        return inputx,labels

#TODO:
# meta-train meta-validation meta-test
# change memory module
    def cmbAndTransImg(self,img,shift,rot, labelOneHot):
        # return np.asarray(np.concatenate(loadTransform(img,shift,rot,self.imgSize), labelOneHot))
        x=loadTransform(img, shift, rot, self.imgSize).flatten()
        y=labelOneHot
        return np.asarray(np.concatenate((x,y)))

# class OmniglotDataset(Dataset):
#     def __init__(self,folderName,
#                  batchSize=16,
#                  nbClsPerEpi=5,
#                  nbSmpsPerCls=10,
#                  size=(20,20),
#                  maxShift=10,
#                  maxSmlRot=15):
#
#         self.imgSize =size
#         self.pixelNum    = self.imgSize[0] * self.imgSize[1]
#         self.batchSize   = batchSize
#         self.nbClsPerEpi = nbClsPerEpi
#         self.nbSmpsPerCls= nbSmpsPerCls
#         self.maxShift    = maxShift
#         self.maxSmlRot   = maxSmlRot
#         self.seqLen      = self.nbSmpsPerCls * self.nbClsPerEpi
#         self.dataAbsPath = os.path.join(os.getcwd(),"data")
#         self.folder      = os.path.join(self.dataAbsPath,folderName)
#         self.fileList    = [os.path.join(self.folder,subfolder,subsubfolder) for subfolder in os.listdir(self.folder) \
#                             for subsubfolder in os.listdir(os.path.join(self.folder,subfolder))]
#         self.inputx = np.zeros((self.batchSize, self.seqLen,self.pixelNum+self.nbClsPerEpi), dtype=float)
#         self.labels = np.zeros((self.batchSize, self.seqLen), dtype=int)
#         random.shuffle(self.fileList)
#
#     def __getitem__(self, idx):
#         if idx==0:
#             self.getSequence()
#         sample = {'image':self.inputx[:,idx,:],'label':self.labels[:,idx]}
#         return sample
#     def __len__(self):
#         return self.seqLen
#
#     def getSequence(self):
#         clsFolders=random.sample(self.fileList,self.nbClsPerEpi) #determine which classes to sample
#         for i in range(self.batchSize):
#         #bigRotate = random.randint(0, 3) * 90
#             bigRotate=0
#             labelsAndImgs = getShuffleImg(clsFolders, self.nbSmpsPerCls)
#             labelsPerBatch, imgs  = zip(*labelsAndImgs)
#             labelsOneHot=np.zeros((self.seqLen,self.nbClsPerEpi), dtype=float)
#             labelsOneHot[0,:]=0
#             labelsOneHot[range(1,self.seqLen), labelsPerBatch[:-1]]=1
#             shifts= np.random.randint(-self.maxShift,self.maxShift,size=(self.seqLen,2))
#             rots  = np.add(np.random.uniform(-self.maxSmlRot,self.maxSmlRot,size=self.seqLen),bigRotate)
#
#             self.inputx[i] = np.asarray([np.concatenate((loadTransform(img,shift,rot+bigRotate,self.imgSize).flatten(), labelOneHot)) \
#                                      for img,shift,rot,labelOneHot in zip(imgs,shifts,rots,labelsOneHot)])
#             self.labels[i]    = np.asarray(labelsPerBatch)
