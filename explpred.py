import numpy as np
import scipy.stats as st
import random
import Orange
import time
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
import copy


class ExplainPredictions(object):
    """

    TODO

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

    def __init__(self, table, model, pError=0.05, error=0.01, batchSize=100, maxIter=10000, minIter=100):

        self.model = model
        self.table = table
        self.pError = pError
        self.error = error
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.minIter = minIter
        self.atr_names = DiscreteVariable(name = 'attributes', values = [var.name for var in table.domain.attributes])





    def anytime_explain(self, instance):

        dataRows, noAtr = self.table.X.shape
        classValue = self.model(instance)[0]
        print(classValue)

        # placeholders : steps, mean, sum of squared differences, calcuated contribuitons
        steps = np.zeros((1, noAtr), dtype=float)
        mu = np.zeros((1, noAtr), dtype=float)
        M2 = np.zeros((1, noAtr), dtype=float)
        expl = np.zeros((1, noAtr), dtype=float)
        var = np.ones((1, noAtr), dtype=float)

        atr_indices = np.asarray(range(noAtr)).reshape((1, noAtr))
        batchMxSize = self.batchSize * noAtr
        zSq = abs(st.norm.ppf(self.pError/2))**2

        tiled_inst = Table.from_numpy(instance.domain, np.tile(instance._x, (self.batchSize, 1)), np.full((self.batchSize, 1), instance._y[0]))
        minReached = np.zeros((1, noAtr), dtype=bool)
        idx = 0
        while not(all(minReached[0,:])):
            if not(any(minReached[0,:])):
                a = np.random.choice(atr_indices[0], p=(var[0,:]/(np.sum(var[0,:]))))
            else:
                a = np.argmin(steps[0,:])
            # as previously
            perm = np.random.choice([True, False], batchMxSize, replace=True)
            perm = np.reshape(perm, (self.batchSize, noAtr))
            rand_data = self.table.X[random.sample(
                range(dataRows), k=self.batchSize), :]
            inst1 = copy.deepcopy(tiled_inst)
            inst1.X[perm] = rand_data[perm]
            inst2 = copy.deepcopy(inst1)

            inst1.X[:, a] = tiled_inst.X[:, a]
            inst2.X[:, a] = rand_data[:, a]
            kk = self.model(inst1)
            f1 = (self.model(inst1) == classValue) * 1
            f2 = (self.model(inst2) == classValue) * 1
            #print (f1-f2)
            diff = np.sum(f1 - f2)
            expl[idx, a] += diff

            # update variance
            steps[0, a] += self.batchSize
            d = diff - mu[0, a]
            mu[0, a] += d/steps[idx, a]
            M2[0, a] += d*(diff - mu[idx, a])
            var[idx, a] = M2[idx, a] / (steps[idx, a]-1)

            # exclude from sampling if necessary
            neededIter = zSq * var[idx, a] / (self.error**2)
            if (neededIter <= steps[idx, a]) and (steps[idx,a] >= self.minIter) or (steps[idx, a] > self.maxIter):
                minReached[0,a] = True
                    
        expl[idx,:] = expl[idx,:]/steps[idx,:]
        #$print (steps)

        #creating return array
        domain = Domain([self.atr_names], [ContinuousVariable('contribuitons')])
        table = Table.from_list(domain, np.asarray(self.atr_names.values).reshape(-1, 1))
        print (steps)
        table.Y = expl.T
        return table


def analyse_xor(num_inst_to_pred=10, printT=False):
    domain = Domain([ContinuousVariable(name = 'x1'), ContinuousVariable(name = 'x2'), \
        ContinuousVariable(name = 'x3'), ContinuousVariable(name = 'x4'), \
        ContinuousVariable(name = 'x5')], 
         [ContinuousVariable(name = 'xor')])
    X = np.random.choice([1, 0], (1000, 5))
    
    Y = (np.logical_xor(X[:, 0], X[:, 1])*1).reshape(-1, 1)
    print (Y.shape)
    data = Table.from_numpy(domain, np.hstack((X,Y)))
    print (data.domain.class_vars)
    #data.Y = Y
    rf = RandomForestLearner()
    model = rf(data)
    e = ExplainPredictions(data, model)
    shapely_any = e.anytime_explain(X[1,:])
    print ("------")

    print (X[1,:])
    print (shapely_any)

def iris():
    data = Orange.data.Table("heart_disease.tab")
    z = 1
    rf = RandomForestLearner()
    model = rf(data)
    e = ExplainPredictions(data, model)
    shapely_any = e.anytime_explain(data[z])
    print ("--------------------")
    print (data.domain.class_vars)
    print (data.X[z,:])
    print (data.Y[z])
    print(shapely_any)

 



#from sklearn.ensemble import RandomForestClassifier
from Orange.modelling import RandomForestLearner
import csv
if __name__ == "__main__":
    #analyse_xor(printT=False)
    iris()
