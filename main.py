import os
import time
import torch
import Dataset
from multiprocessing import Pool
from Model import  NTMModel
from torch.autograd import Variable
from torch import nn
from torch import optim

Args={
    'EpisodeNum':10000,
    'Nway':5,
    'Kshot':1,
    'ImgSize':(20, 20)}

def loadDataSet(samples):

    omniglotDS = Dataset.OmniglotDataset(folderName="train", size=Args['ImgSize'])
    p = Pool(processes=8)
    iterDS = iter(omniglotDS)

    print("Start loading data")
    for i in range(Args['EpisodeNum']):
        samples.append(p.apply_async(next, (iterDS,)))


def train(samples):
    #TODO: use packedSequence
    print("Start training")

    inputDim = Args['ImgSize'][0]**2 + Args['Nway'] #Assumption:One-Hot vec
    t0 = time.time()
    criterion=nn.CrossEntropyLoss()
    NTM = NTMModel(Args['Nway'], inputDim, _BatchSize=16)
    NTM.cuda()
    optimizer=optim.RMSprop(NTM.parameters(),lr=5e-1,weight_decay=0.95,momentum=0.9)
    running_loss=0
    for sample in samples:
        i, (x, label) = sample.get()
        x, label = torch.from_numpy(x), torch.from_numpy(label)
        x, label = Variable(x.float().cuda()), Variable(label.long().cuda())
        optimizer.zero_grad()
        yEsti=NTM(x)
        # loss=criterion(yEsti,label)
        # loss.backward()
        for celli in range(len(yEsti)):
            loss = criterion(yEsti[celli],label[celli])
            loss.backward()
        optimizer.step()
        running_loss+=loss.data[0]
        if i%20==0:
            print('time:%.1f [Episodes:%d] loss:%.3f' % (time.time()-t0,i+1, running_loss/20))
            t0=time.time()
            running_loss=0.0
    print("Finished training")
def main(projPath):
    #TODO: Modify the program to match the training condition and testing condition
    # omniglotDS = Dataset.OmniglotDataset(folderName="train", size=Args['ImgSize'])
    #load dataset
    samples=[]
    t0=time.time()
    try:
        loadDataSet(samples)
        train(samples)

    except KeyboardInterrupt:
        print(time.time() - t0)


if __name__ == "__main__":
    #TODO: Next run program on graphical card
    projPath = os.getcwd()
    print(projPath)
    main(projPath)