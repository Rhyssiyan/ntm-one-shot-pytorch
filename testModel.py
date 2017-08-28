import torch
import numpy as np
import sys

def testAccu(NTM, batchSize=16, seqLen=50):
    file = 'test_k1_N5.npy.npz'
    npzfile = np.load(file)
    testx = npzfile['testx']
    labels = npzfile['testy']  # seqLen*batchSize ndarray
    labels = torch.from_numpy(labels)
    pred = NTM(testx)  # seqLen*batchSize*Nway
    _, pred = torch.max(pred, dim=2)  # get index
    correctN = torch.sum(torch.eq(pred, labels))
    accu = correctN / (batchSize * seqLen)
    accu=0
    return accu


def testAccuInTraining(pred, label, nbCls):
    """

    :param pred:seqLen*batchSize*nbClass list+floatTensor
    :param label: batchSize*nbClass
    :return:
    """
    for i,p in enumerate(pred):
        pred[i]=torch.unsqueeze(p,dim=0)
    pred=torch.cat(pred,dim=0)
    _, pred = torch.max(pred, dim=2)  # get index
    correctN=torch.sum(torch.eq(pred, label),dim=0)
    indLst=range(nbCls-1,len(correctN),nbCls)
    return [correctN[ind] for ind in indLst]
