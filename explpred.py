import numpy as np
import scipy.stats as st
import random
import time


class ExplainPredictions(object):
    """
    Class used to explain individual predictions by determining the importance of attribute values. All interactions between atributes are accounted for by calculating Shapely value.
    Parameters:

    model : 
        model to be used with classification, usinng predict method
    data : matrix
        dataset without target values
    target: 
        target values of given dataset
    error : float
        desired error 
    pError : float
        p value of error
    batchSize : int
        size of batch to be used in classification, bigger batch size speeds up the calculations and improves estimations of variance
    maxIter : int
        maximum number of iterations per attribute
    minIter : int
        minimum number of iterations per attirubte

    Return: 
        array dimensions numInstancesToPredict * numAttributes. Field i,j is Shapely value for i-th instance, value of j-th atribute.

    """



    def __init__(self, model, data, target, pError= 0.01, error = 0.01, batchSize = 50, maxIter = 20000, minIter = 1000):
        
        self.model = model
        self.data = data
        self.target = target
        self.pError = pError
        self.error = error
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.minIter = minIter

    def prepare(self, instances):
        """
        Changes instance shape if needed, removes rows with missing values
    
        """
        if not instances.size:
            raise ValueError("Instance is empty.")
        elif instances.ndim == 1:
            instances = np.reshape(instances,(1,-1))
        instances = instances[~np.isnan(instances).any(axis=1)]

        return instances
 


    def set_data(self, data, target):
        self.data = data
        self.target = target

    def set_params(self, pError = 0.01, error = 0.01, batchSize = 50, maxIter = 10000, minIter = 1000):
        self.pError = pError
        self.error = error
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.minIter = minIter


    def anytime_explain(self, instances):
    
        instances = self.prepare(instances)
        noClasses = np.unique(self.target).size 
        dataRows, noAtr = self.data.shape
        numInst = len(instances)
    
        #placeholders : steps, mean, sum of squared differences, calcuated contribuitons
        steps = np.zeros((1, noAtr), dtype = float)
        mu = np.zeros((1, noAtr), dtype = float)
        M2 = np.zeros((1, noAtr), dtype = float)
        expl = np.zeros((numInst, noAtr), dtype = float)
        var = np.ones((1, noAtr), dtype = float)
        atr_indices = np.ones((1, noAtr), dtype = int)

        atr_indices[0] = np.asarray(range(noAtr))
        tiled_inst = np.tile(instances,(self.batchSize,1))
        batchMxSize = self.batchSize * noAtr
        zSq = abs(st.norm.ppf(self.pError/2))**2
        i = 0

        while any(steps[0] < self.minIter):
            if (all(steps[0] < self.minIter)):
                a = np.random.choice(atr_indices[0], p=(var[0]/(sum(var[0]))))
            else :
                a = np.argmin(steps)
            #as previously
            perm = np.random.choice([True, False], batchMxSize, replace=True)
            perm = np.reshape(perm, (self.batchSize, noAtr)) 
            rand_data = self.data[random.sample(range(dataRows), k=self.batchSize),:]
            inst1 = np.copy(tiled_inst)
            inst1[perm] = rand_data[perm]
            inst2 = np.copy(inst1)

            #inst2 = np.copy(tiled_inst)
            #inst2[perm] = rand_data[perm]
            inst1[:, a]  = tiled_inst[:,a]
            inst2[:, a] = rand_data[:, a]
                
            f1 = self.model.predict(inst1)
            f2 = self.model.predict(inst2)
            diff = sum(f1 - f2)
            expl[i,a] += diff

            #update variance
            steps[i,a] += self.batchSize
            d  = diff - mu[i,a]
            mu[i,a] += d/steps[i,a]
            M2[i,a] += d*(diff - mu[i,a])
            var[i,a] = M2[i,a] / (steps[i,a]-1)  

            #exclude from sampling if necessary
            neededIter = zSq * var[i,a]/ (self.error**2)
            if neededIter <= steps[i,a]:
               steps[i,a] = self.minIter

        expl = expl/steps

        return expl





def analyse_xor(num_inst_to_pred = 10, printT = False):
    X = np.random.choice([1,0], (10000,2))
    Y = np.logical_xor(X[:,0], X[:,1])*1
    clf = RandomForestClassifier(max_depth=10, max_features = None)
    model = clf.fit(X,Y)
    e = ExplainPredictions(model, X, Y)
    ofile = open("./xor_est.csv", "w")
    writer = csv.writer(ofile) 
    shapely_any = e.anytime_explain(np.asarray([1,1]))

    print (shapely_any)
    if printT:
        for row in shapely_v:
            writer.writerow(row)
        ofile.close()
        

    
from sklearn.ensemble import RandomForestClassifier   
import csv
if __name__ == "__main__":
        analyse_xor(printT=False)






