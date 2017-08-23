import os
import time

# from src import Dataset
import Dataset

config={}

def main(projPath):
    omniglotDS = Dataset.OmniglotDataset(folderName="train", projAbsPath=projPath)
    t0 = time.time()
    try:
        for i,(inputx, label) in omniglotDS: #try don't put (inxx,label)
            print(time.time()-t0,"num: ",i)
            t0=time.time()
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