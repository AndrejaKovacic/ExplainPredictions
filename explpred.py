import numpy as np
import scipy.stats as st
import random
import Orange
import time
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.base import Model
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

    def __init__(self, data, model, pError=0.05, error=0.05, batchSize=200, maxIter=50000, minIter=100):

        self.model = model
        self.data = data
        self.pError = pError
        self.error = error
        self.batchSize = batchSize
        self.maxIter = maxIter
        self.minIter = minIter
        self.atr_names = DiscreteVariable(name = 'attributes', values = [var.name for var in data.domain.attributes])





    def anytime_explain(self, instance):

        dataRows, noAtr = self.data.X.shape
        classValue = self.model(instance)[0]

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
        inst1 = copy.deepcopy(tiled_inst)
        inst2 = copy.deepcopy(tiled_inst)
        iterations_reached = np.zeros((1, noAtr))

        while not(all(iterations_reached[0,:] > self.maxIter)):
            if not(any(iterations_reached[0,:] > self.maxIter)):
                a = np.random.choice(atr_indices[0], p=(var[0,:]/(np.sum(var[0,:]))))
            else:
                a = np.argmin(iterations_reached[0,:])

            perm = np.random.choice([True, False], batchMxSize, replace=True)
            perm = np.reshape(perm, (self.batchSize, noAtr))
            rand_data = self.data.X[random.sample(
                range(dataRows), k=self.batchSize), :]
            inst1.X = np.copy(tiled_inst.X)
            inst1.X[perm] = rand_data[perm]
            inst2.X = np.copy(inst1.X)

            inst1.X[:, a] = tiled_inst.X[:, a]
            inst2.X[:, a] = rand_data[:, a]
            f1 = self._get_predictions(inst1, classValue)
            f2 = self._get_predictions(inst2, classValue)
            
            diff = np.sum(f1 - f2)
            expl[0, a] += diff

            # update variance
            steps[0, a] += self.batchSize
            iterations_reached[0,a] += self.batchSize
            d = diff - mu[0, a]
            mu[0, a] += d/steps[0, a]
            M2[0, a] += d*(diff - mu[0, a])
            var[0, a] = M2[0, a] / (steps[0, a]-1)

            # exclude from sampling if necessary
            neededIter = zSq * var[0, a] / (self.error**2)    
            if (neededIter <= steps[0, a]) and (steps[0,a] >= self.minIter) or (steps[0, a] > self.maxIter):
                iterations_reached[0,a] = self.maxIter + 1
                
                    
        expl[0,:] = expl[0,:]/steps[0,:]

        #creating return array
        domain = Domain([self.atr_names], [ContinuousVariable('contribuitons')])
        table = Table.from_list(domain, np.asarray(self.atr_names.values).reshape(-1, 1))
        print (steps)
        table.Y = expl.T
        return classValue, table

    def _get_predictions(self, inst, classValue):
        if isinstance(self.data.domain.class_vars[0], ContinuousVariable):
            #regression
            return self.model(inst)
        else:
            #classification
            predictions =  (self.model(inst) == classValue) * 1
            #return self.model(inst, Model.ValueProbs)[1][:,int(classValue)]
            return predictions 



def analyse_xor(num_inst_to_pred=10, printT=False):
    domain = Domain([ContinuousVariable(name = 'x1'), ContinuousVariable(name = 'x2')
        , \
        ContinuousVariable(name = 'x3'), ContinuousVariable(name = 'x4'), \
        ContinuousVariable(name = 'x5')
        , ContinuousVariable(name = 'x6'), ContinuousVariable(name = 'x7'), \
        ContinuousVariable(name = 'x8'), ContinuousVariable(name = 'x9'), \
        ContinuousVariable(name = 'x10')], 
         [ContinuousVariable(name = 'xor')])
         #[DiscreteVariable(name = 'xor', values = ['0','1'])])
    X = np.random.choice([1, 0], (1000, 10))
    
    Y = (np.logical_xor(X[:, 0], X[:, 1])*1).reshape(-1, 1)
    Y = np.char.mod('%d', Y)
    print (Y.shape)
    data = Table.from_numpy(domain, np.hstack((X,Y)))
    print (data.domain.class_vars)
    #data.Y = Y
    rf = RandomForestLearner()
    model = rf(data)
    e = ExplainPredictions(data, model)
    print ("------")
    shapely_any = e.anytime_explain(data[1])

    print (data[1])
    print (shapely_any)

def test(name):
    data = Orange.data.Table(name+".tab")
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




 


from Orange.modelling import RandomForestLearner
import sys
if __name__ == "__main__":
    #iris, heart_disease, servo
    target = sys.argv[1]
    if target == 'xor':
        analyse_xor(printT=False)
    else:
        test(target)
