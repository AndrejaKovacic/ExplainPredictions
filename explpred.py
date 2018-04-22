import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as st
import random
from Orange.data import Table


class ExplainPredictions(object):
    """
    Class used to explain individual predictions made on our data. All interactions between atributes are accounted by calculating Shapely value.
    Parameters:

    data : matrix
    orange dataset with attribute X, which transforms (if needed) attributes to numeric and discretizes them


    """

    #TODO : change so it predicts for multiple instances


    def __init__(self, model, data, target, pError= 0.05, error = 0.05, batchSize = 3, maxIter = 1000):
        
        self.model = model
        self.data = data.X
        self.target = target
        self.pError = pError
        self.error = error
        self.batchSize = batchSize
        self.maxIter = maxIter


    def prepare(self, instances):
        """
        Changes instance shape if needed,  TODO : handle continuous attributes, missing values
    
        """
        if not instances.size:
            raise ValueError("Instance is empty.")
        elif instances.ndim == 1:
            instances = np.reshape(instances,(1,-1))



        return instances
 

    def shuffle(self, array, axis = 1):
        """
        Shuffles array along specified axis
        """
        a = np.random.random(array.shape)
        idx = np.argsort(a, axis)
        return array[np.arange(a.shape[0])[:,None],idx]


    def expl_one_atr(self, instances):
        """
        Calculates which attributes are most important in classification of given instance, one attribute at the time.

        Parameters: 
        instances - array of instances for which we want to obtain predictions
        """
        instances = self.prepare(instances)
        noClasses = np.unique(self.target).size 
        dataRows, noAtr = self.data.shape
        numInst = len(instances)
    
        #placeholders
        expl = np.zeros((numInst, noAtr), dtype = float)
        stddev = np.zeros((numInst, noAtr), dtype = float)
        itr = np.zeros((numInst, noAtr), dtype = float)
        perm = np.zeros((dataRows, noAtr), dtype=bool)
    
        batchMxSize = self.batchSize * noAtr
        #absolute value of z score of p-th quantile in normal distribution
        zSq = abs(st.norm.ppf(self.pError/2))**2
        errSq = self.error**2
        
        for i in range(numInst):
            inst = instances[i, :]   
            for a in range(noAtr):
                noIter = 0
                moreIterations = True
                while(moreIterations):
                    perm = np.random.choice([True, False], batchMxSize, replace=True)
                    perm = np.reshape(perm, (self.batchSize, noAtr)) 
                    #not to be replaced
                    perm[:,a] = False
                    #Semi random data - atributes data is first permuted in columns, then we sample desired number of rows
                    inst1 = self.shuffle(self.data)[random.sample(range(dataRows), k=self.batchSize),:]
                    inst1[perm] = np.tile(inst,(self.batchSize,1))[perm]
                    #inst2 has all succeding (including a-th) atributes filled with random values
                    inst2 = np.copy(inst1) 
                    #inst1 has a-th atribute filled with selected instance
                    inst1[:,a] = np.tile(inst, (self.batchSize,1))[:,a]
                
                    f1 = model.predict(inst1)
                    f2 = model.predict(inst2)
                    diff = f1- f2
                    expl[i, a] = expl[i, a] + sum(diff)
                    noIter = noIter + self.batchSize
                    stddev[i, a] = stddev[i, a] + diff.dot(diff)
                    v2 = stddev[i, a]/noIter - (expl[i, a]/noIter)**2
                    neededIter = zSq * v2 / errSq
                    if (neededIter <= noIter or noIter > self.maxIter):
                        moreIterations = False
    
                expl[i, a] = expl[i, a] / noIter
                stddev[i, a] = np.sqrt(stddev[i, a] / noIter - expl[i, a]/noIter**2)
                itr[i, a] = noIter
    
        return expl 
        
    def exhaustive_shapely(self, instances):
        instances = self.prepare(instances)
        noClasses = np.unique(self.target).size 
        dataRows, noAtr = self.data.shape
        numInst = len(instances)
    
        #placeholders
        expl = np.zeros((numInst, noAtr), dtype = float)
        stddev = np.zeros((numInst, noAtr), dtype = float)
        itr = np.zeros((numInst, noAtr), dtype = float)
        perm = np.zeros((dataRows, noAtr), dtype=bool)



iris = datasets.load_iris()
X = iris.data
y = iris.target
trainX, testX = X[:145], X[145:]

trainY, testY = y[:145], y[145:]
clf = RandomForestClassifier(max_depth=3, random_state=0)
model = clf.fit(trainX, trainY)

e = ExplainPredictions(model, trainX, trainY)
print(e.expl_one_atr(X[145]))
#print (e.expl_all(X[145]))