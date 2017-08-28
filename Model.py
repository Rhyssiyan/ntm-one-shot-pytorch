import torch.nn as nn
import torch
import torch.tensor
from torch.autograd import  Variable
class NTMModel(nn.Module):

    def __init__(self, _Nway, _InputDim, _BatchSize, _GPUID):
        super(NTMModel, self).__init__()
        # arguments:
        self.decayGamma=0.99
        self.Nway=_Nway
        self.batchSize=_BatchSize
        self.inputDim=_InputDim
        self.hiddDim = 200
        self.memSize = (128,40) # slots* size
        self.nbRead  = 4

        #TODO: try putting r_t into input

        self.ctrler  = nn.LSTMCell(self.inputDim, self.hiddDim)

        self.outFc   = nn.Sequential(nn.Linear(self.hiddDim+self.nbRead*self.memSize[1], self.Nway), nn.Sigmoid())

        self.keyFc   = nn.Sequential(nn.Linear(self.hiddDim, self.nbRead*self.memSize[1]), nn.Tanh())
        self.sigmaFc = nn.Sequential(nn.Linear(self.hiddDim, self.nbRead*1), nn.Tanh(), nn.Sigmoid())

        self.softM  = nn.Softmax()

        # Variable default requires_grad=False
        self.M_tm1  = Variable(torch.FloatTensor(self.batchSize, self.memSize[0], self.memSize[1])).cuda(_GPUID)
        self.h_tm1  = Variable(torch.FloatTensor(self.batchSize, self.hiddDim)).cuda(_GPUID)
        self.c_tm1  = Variable(torch.FloatTensor(self.batchSize, self.hiddDim)).cuda(_GPUID)

        self.wr_tm1 = Variable(torch.FloatTensor(self.batchSize, self.nbRead, self.memSize[0])).cuda(_GPUID)
        self.wu_tm1 = Variable(torch.FloatTensor(self.batchSize, self.memSize[0])).cuda(_GPUID)
        self.wlu_tm1= Variable(torch.FloatTensor(self.batchSize, self.memSize[0])).cuda(_GPUID)



        # nn.init.xavier_uniform(self.outFc[0].weight)
        # self.outFc[0].bias.data.zero_()
        #
        # nn.init.xavier_uniform(self.keyFc[0].weight)
        # self.keyFc[0].bias.data.zero_()
        #
        # nn.init.xavier_uniform(self.sigmaFc[0].weight)
        # self.outFc[0].bias.data.zero_()

        #TODO: add a_t
        #      put r_t into input

    def forward(self,x):
        # x: seqLen * batch_size * inputVec(imgVec + oneHot)
        # clear
        self.wr_tm1.data.zero_()
        self.wu_tm1.data.zero_()
        nn.init.constant(self.wlu_tm1,1)
        self.M_tm1.data.zero_()
        self.h_tm1.data.zero_()
        self.c_tm1.data.zero_()
        # y=Variable(torch.FloatTensor())
        y=[]
        for i in range(len(x)):

            self.h_t, self.c_t=self.ctrler(x[i], (self.h_tm1,  self.c_tm1))# LSTM Cell  x[i]:batch_size*(inputVec+oneHot) h_tm1:batchSize*hiddDim

            # TODO: why use h_t not c_t
            # erase -> write -> read
            # erase
            #find the minimum of wu_tm1
            _,ind=torch.min(self.wu_tm1.data,1) # wu_tm1:batchSize * slots
            self.M_tm1.data[range(self.batchSize), ind, :]=0

            # write
            self.k_t = self.keyFc(self.h_t) # Variable
            self.k_t = self.k_t.view(self.batchSize, self.nbRead, self.memSize[1])  # batchSize * nbRead * memSize[1]

            self.sigma_t= self.sigmaFc(self.h_t)
            self.sigma_t= self.sigma_t.view(self.batchSize, self.nbRead, 1)

            #get ww_t: batchSize*nbRead*slots Parameter
            self.ww_t   = self.sigma_t * self.wr_tm1   # batchSize*nbRead*1    batchSize*1*slots
            self.ww_t   += torch.bmm((1-self.sigma_t), torch.transpose(torch.unsqueeze(self.wlu_tm1,2),2,1))
            #update memory
            self.M = self.M_tm1 + torch.bmm(torch.transpose(self.ww_t,2,1) , self.k_t)  # M:batchSize*slots*size ww_t:batchSize*nbRead*slots k_t:batchSize*nbRead*size

            # read
            # calc cos similarity between k_t and M_i
            self.K=torch.bmm(self.k_t, torch.transpose(self.M,2,1))   # k_t:batchSize * nbRead * size(i.e. memSize[1]) * trans(M): batchSize*size*slots
            # K:batchSize*nbRead*slots
            normk_t = torch.unsqueeze(torch.norm(self.k_t, p=2, dim=2),2)   # k_t:batchSize * nbRead * memSize[1] =>  batchSize*nbRead*1
            normM_t = torch.unsqueeze(torch.norm(self.M, p=2, dim=2),1)# M:batchSize*slots*size              => normM_t batchSize*1*slots
            dominator=torch.bmm(normk_t, normM_t) #batchSize*nbRead*slots
            self.K=self.K/dominator

            # calc wr_t
            self.K=self.K.view(self.batchSize*self.nbRead,-1) #(batchSize*nbRead)*slots
            self.wr_t=self.softM(self.K)
            self.wr_t=self.wr_t.view(self.batchSize,self.nbRead,-1)#batchSize*nbRead*slots
            # get r_t
            self.r_t = torch.bmm(self.wr_t,self.M)
            self.r_t = self.r_t.view(self.batchSize,-1) #batchSize*(nbRead*size)

            #TODO: It's only the result at time i
            ntm_out_i = self.outFc(torch.cat([self.h_t,self.r_t],dim=1))
            ntm_out_i = self.softM(ntm_out_i) #batchSize*nbClsPerEpi
            y.append(ntm_out_i)
            # ntm_out_i = torch.unsqueeze(ntm_out_i, dim=0)
            # if i==0:
            #     y=ntm_out_i
            # else:
            #     y=torch.cat([y,ntm_out_i], dim=0)

            # update weights for time t
            self.wr_tm1 = self.wr_t.detach()
            self.ww_tm1 = self.ww_t.detach() #If don't add detach what is the result
            self.wu_tm1   = self.decayGamma*self.wu_tm1 + torch.sum(self.wr_tm1 + self.ww_tm1, dim=1) #batch*slots
            self.topkVal,self.topkInd= torch.topk(self.wu_tm1, k=self.nbRead, dim=1, largest=False) #batch*1se
            self.kthSmall=self.topkVal[:,-1]
            self.wlu_tm1 = torch.le(self.wu_tm1, torch.unsqueeze(self.kthSmall,dim=1)).float().detach()
            self.M_tm1 = self.M.detach()
        return y





