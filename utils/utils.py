import random
import numpy as np


from sklearn.model_selection import train_test_split


def generateTrimIndex(arrSize) :
    import random
    n = random.randint(0,100000)
    np.random.seed(n)

    ##print(arrSize)
    indices = np.arange(0, arrSize, 1)
    ##print(indices.shape)
    filteredIdx = np.random.choice(indices,10,replace=False)
    return filteredIdx

def trimNoOfFrames(facialLMArr):
    numFeatures = facialLMArr.shape[0]
    print('trimNoOfFrames : numFeatures = ',numFeatures)
    filteredIdx = generateTrimIndex(numFeatures)
    print(filteredIdx)
    print(facialLMArr[filteredIdx, :])

    return facialLMArr[filteredIdx, :]

def trimFeatures(x) :
    trimX = []
    numFeaturesDataSet = x.shape[0]
    print('numFeaturesDataSet == ', numFeaturesDataSet)
    trimX = np.array(trimNoOfFrames(x))

    print('trimX.shape == ',trimX.shape)
    return trimX



def convertToNumpyArr( rawFP ) :
    tmp = rawFP.replace(' ', '').replace('[ ', '').replace(',[', '').replace('\n', '').replace(';', '').replace('[', '').split(']')
    tmpList = []
    for i in tmp :
        t = i.split(',')
        if( len(t) == 136 ) :
            #print(t)
            x =  np.array(t)
            y =  x.astype(np.float)
            tmpList.append(y)

    tmpArr = np.array(tmpList)
    print(tmpArr.shape)
    trimX = trimFeatures(tmpArr)
    print(trimX.shape)

    return trimX

