import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as st
import random


def prepare(inst):
    """
    Changes instance shape if needed,  TODO : handle continuous atributes, missing values

    """
    if not inst.size:
        raise ValueError("Instance is empty.")
    elif inst.ndim == 1:
        inst = np.reshape(inst,(1,-1))

    if np.size(inst,0)  != 1:
        raise ValueError("First dimension of instance is too big. Expected dimension = 1.")  

    return inst  

def shuffle(array, axis = 1):
    """
    Shuffles array along specified axis
    """
    a = np.random.random(array.shape)
    idx = np.argsort(a, axis)
    return array[np.arange(a.shape[0])[:,None],idx]


def expl_one_atr(model, inst, data, target, pError= 0.05, error = 0.05, batchSize = 3, maxIter = 1000):
    """
    Calculates which atributes are most important in classification of given instance.

    """
    inst = prepare(inst)
    noClasses = np.unique(target).size 
    noInst, noAtr = data.shape

    #placeholders
    expl = np.zeros(noAtr, dtype = float)
    stddev = np.zeros(noAtr, dtype = float)
    itr = np.zeros(noAtr, dtype = float)
    perm = np.zeros((noInst, noAtr), dtype=bool)

    batchMxSize = batchSize * noAtr
    #absolute value of z score of p-th quantile in normal distribution
    zSq = abs(st.norm.ppf(pError/2))**2
    errSq = error**2

    for a in range(noAtr):
        noIter = 0
        moreIterations = True
        while(moreIterations):
            perm = np.random.choice([True, False], batchMxSize, replace=True)
            perm = np.reshape(perm, (batchSize, noAtr)) 
             #not to be replaced
            perm[:,a] = False
            #Semi random data - atributes data is first permuted in columns, then we sample desired number of rows
            inst1 = shuffle(data)[random.sample(range(noInst), k=batchSize),:]
            inst1[perm] = np.tile(inst,(batchSize,1))[perm]
            #inst2 has all succeding (including a-th) atributes filled with random values
            inst2 = np.copy(inst1) 
            inst1[:,a] = np.tile(inst, (batchSize,1))[:,a]
        
            f1 = model.predict(inst1)
            f2 = model.predict(inst2)
            diff = f1- f2

            expl[a] = expl[a] + sum(diff)
            noIter = noIter + batchSize
            stddev[a] = stddev[a] + diff.dot(diff)
            v2 = stddev[a]/noIter - (expl[a]/noIter)**2
            neededIter = zSq * v2 / errSq
            if (neededIter <= noIter or noIter > maxIter):
                moreIterations = False

        expl[a] = expl[a] / noIter
        stddev[a] = np.sqrt(stddev[a] / noIter - expl[a]/noIter**2)
        itr[a] = noIter

    print (expl) 
    print (itr)   


iris = datasets.load_iris()
X = iris.data
y = iris.target
trainX, testX = X[:145], X[145:]

trainY, testY = y[:145], y[145:]
clf = RandomForestClassifier(max_depth=3, random_state=0)
model = clf.fit(trainX, trainY)

expl_one_atr(model, X[145],  trainX, y[145])