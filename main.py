import os
import time
import testModel
import torch
import Dataset
import multiprocessing
from multiprocessing import Pool#,Queue
from multiprocessing.queues import SimpleQueue
from Model import  NTMModel
from torch.autograd import Variable
from torch import nn
from torch import optim
# from asyncio import Queue
import objgraph
from queue import Queue
Args={
    'EpisodeNum':50000,
    'Nway':5,
    'Kshot':1,
    'ImgSize':(20, 20),
    'nbClsPerEpi':5,
    'nbSmpsPerCls':10,
    'batchSize':16}
# @profile
def loadDataSet(samples):

    omniglotDS = Dataset.OmniglotDataset(folderName="train", size=Args['ImgSize'],batchSize=Args['batchSize'],nbClsPerEpi=Args['nbClsPerEpi'],nbSmpsPerCls=Args['nbSmpsPerCls'])
    Args['SeqLen']=Args['nbClsPerEpi']*Args['nbSmpsPerCls']
    p = Pool(processes=16)
    iterDS = iter(omniglotDS)
    x,y=omniglotDS.generateSmallTestSet(kShot=1,Nway=5)
    print("Start loading data")
    for i in range(Args['EpisodeNum']):
        samples.put(p.apply_async(next, (iterDS,)))
        # samples.append(p.apply_async(next, (iterDS,)))

# @profile
def train(samples):
    #TODO: use packedSequence
    print("Start training")
    GPUID = 1
    inputDim = Args['ImgSize'][0]**2 + Args['Nway'] #Assumption:One-Hot vec
    t0 = time.time()
    criterion=nn.CrossEntropyLoss().cuda(GPUID)
    NTM = NTMModel(Args['Nway'], inputDim, _BatchSize=16, _GPUID=GPUID)
    NTM.cuda(GPUID)
    optimizer=optim.RMSprop(NTM.parameters(),lr=5e-1,weight_decay=0.95,momentum=0.9)
    running_loss=0
    i=0
    accuDir={}
    while samples.empty() is not None:
        i+=1
        sample=samples.get()
    # for i,sample in enumerate(samples):
        _, (x, label) = sample.get() #problem : i is always 1
        x, label = torch.from_numpy(x), torch.from_numpy(label) # labels:0-4
        x, label = Variable(x.float().cuda(GPUID)), Variable(label.long().cuda(GPUID))
        optimizer.zero_grad()
        yEsti=NTM(x)
        # loss=criterion(yEsti,label)
        # loss.backward()
        for celli in range(len(yEsti)):
            loss = criterion(yEsti[celli],label[celli])
            loss.backward()
        optimizer.step()
        running_loss+=loss.data[0]
        # print("count:{0}".format(i))

        if i%20==19:
            print('time:%.1f [Episodes:%d] loss:%.3f' % (time.time()-t0,i, running_loss/20))
            t0=time.time()
            running_loss=0.0
            # objgraph.show_most_common_types()
        if i%100==99:
            # accu=testModel.testAccu(NTM,batchSize=Args['batchSize'],seqLen=Args['SeqLen'])
            accuArr = testModel.testAccuInTraining(yEsti,label, Args['nbClsPerEpi'])
            print('Accuracy:',["{0}st accu:{1}".format(i,accu) for i,accu in enumerate(accuArr)])
            accuDir['{0}'.format(i)]=accuArr
        if i%10000==9999:
            torch.save({'Episodes':i,
                        'state_dict':NTM.state_dict(),
                        'accu':accuDir},'ckpts/checkpoint{0}.tar'.format(i))
    print("Finished training")

def main():
    #TODO: Modify the program to match the training condition and testing condition
    # omniglotDS = Dataset.OmniglotDataset(folderName="train", size=Args['ImgSize'])
    #load dataset
    samples=Queue()
    # samples=[]
    # manager=multiprocessing.Manager()
    # samples=manager.Queue()
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
    main()
