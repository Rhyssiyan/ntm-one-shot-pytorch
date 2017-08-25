import os
import time

# from src import Dataset
import Dataset
from multiprocessing import Pool
config={}
def getNext(iterDS):
    return next(iterDS)
def main(projPath):
    omniglotDS = Dataset.OmniglotDataset(folderName="train", projAbsPath=projPath)
    t0 = time.time()
    iterDs=iter(omniglotDS)
    try:
        p=Pool(processes=8)
        iterDS=iter(omniglotDS)
        resList=[]
        for i in range(100):
        # for i,(inputx, label) in omniglotDS: #try don't put (inxx,label)
        #     resList.append(p.apply_async(getNext,(iterDS,)))
            resList.append(p.apply_async(getNext, (iterDS,)))
            # print(time.time()-t0,"num: ",i)
            # t0=time.time()
        print(time.time() - t0, "num: ", i)
        p.close()
        p.join()
        print(resList)
    except KeyboardInterrupt:
        print(time.time() - t0)

    # omniglotDS = Dataset.OmniglotDataset('train')
    # dataloader=DataLoader(omniglotDS,num_workers=8)
    # for j in range(100):
    #     t0 = time.time()
    #     for i,sample in enumerate(dataloader): #try don't put (inxx,label)
    #         tmp=1
    #     print(time.time()-t0,"seq num: ",j)

if __name__ == "__main__":

    projPath = os.getcwd()
    print(projPath)
    main(projPath)