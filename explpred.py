import numpy as np
import scipy.stats as st
import random



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



    def __init__(self, model, data, target, pError= 0.01, error = 0.01, batchSize = 50, maxIter = 10000, minIter = 1000):
        
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
        batchMxSize = self.batchSize * noAtr
        #absolute value of z score of p-th quantile in normal distribution
        zSq = abs(st.norm.ppf(self.pError/2))**2

        errSq = self.error**2
        
        for i in range(numInst):
            inst = instances[i, :]
            tiled_inst = np.tile(inst,(self.batchSize,1))
            for a in range(noAtr):
                noIter = 0
                moreIterations = True
                while(moreIterations ):
                    perm = np.random.choice([True, False], batchMxSize, replace=True)
                    perm = np.reshape(perm, (self.batchSize, noAtr)) 
                    rand_data = self.data[random.sample(range(dataRows), k=self.batchSize),:]
                    inst1 = np.copy(tiled_inst)
                    inst1[perm] = rand_data[perm]
                    inst1[:, a]  = tiled_inst[:,a]
                    inst2 = np.copy(tiled_inst)
                    inst2[perm] = rand_data[perm]
                    inst2[:, a] = rand_data[:, a]
                
                    f1 = self.model.predict(inst1)
                    f2 = self.model.predict(inst2)
                    diff = f1 - f2

                    expl[i, a] += sum(diff)
                    noIter += self.batchSize
                    stddev[i, a] += diff.dot(diff)
                    v2 = stddev[i, a]/noIter - (expl[i, a]/noIter)**2
                    neededIter = zSq * v2 / errSq
                    if (not noIter < self.minIter and (neededIter <= noIter or noIter > self.maxIter)):
                        moreIterations = False
                #print ("num iter " +  str(noIter))
                #print ("expained " + str(inst[a]))
                #print ("needed " + str(neededIter))
                #print (expl[i,a] / noIter)
                expl[i, a] /= noIter 
                stddev[i, a] = np.sqrt(stddev[i, a] / noIter - expl[i, a]/noIter**2)

    
        return expl 


def analyse_xor(num_inst_to_pred = 10, print = False):
    X = np.random.choice([1,0], (10000,2))
    Y = np.logical_xor(X[:,0], X[:,1])*1
    clf = RandomForestClassifier(max_depth=10, max_features = None)
    model = clf.fit(X,Y)
    e = ExplainPredictions(model, X, Y)
    ofile = open("./xor_est.csv", "w")
    writer = csv.writer(ofile)
    shapely_v = e.expl_one_atr(X[:num_inst_to_pred])
    if print:
        for row in shapely_v:
            writer.writerow(row)
        ofile.close()
        

    
from sklearn.ensemble import RandomForestClassifier   
import csv
if __name__ == "__main__":
        analyse_xor(print=True)






