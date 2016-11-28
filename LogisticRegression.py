import scipy.optimize
import numpy
import scipy
import random
import sklearn.datasets
import sklearn.linear_model



class LogisticRegression:
    def __init__(self, dataX, dataY, iterations=100, learningRate=0.0001):
        self.numSamples = dataX.shape[0]
        self.numParams = dataX.shape[1]
        self.dataX = dataX
        self.dataY = dataY
        self.iterations = iterations
        self.learningRate = learningRate

                
    @staticmethod
    def _loss(params, dataX, dataY):
        # computes binomial deviance, 
        # *** predictions are log odds ***
        pred = numpy.dot(dataX, params)
        pred = pred.ravel()
        return -2.0 * numpy.mean(dataY * pred - numpy.logaddexp(0.0, pred))

    @staticmethod      
    def _gradient(params, dataX, dataY):
        # gradient of binomial deviance loss,
        # *** predictions are log odds ***\
        pred = numpy.dot(dataX, params)
        #pred = pred.ravel()
        #ret = dataY - scipy.special.expit(pred.ravel())
        ret = numpy.dot(dataX.T, (scipy.special.expit(pred) - dataY))/dataX.shape[0]       
        return ret

    @staticmethod
    def printParams(params):
        print(params)

    
    def fit(self):
        # initialize params to random
        self.modelParams = numpy.array([random.randint(-100,100) for i in range(self.numParams)])
        self.optimLogit  = scipy.optimize.fmin_bfgs(LogisticRegression._loss, x0 = self.modelParams,\
           args = (self.dataX, self.dataY), gtol = 1e-4, fprime = LogisticRegression._gradient, callback = LogisticRegression.printParams)
        pass
            

    def predict(self, dataX):
        ret =  numpy.dot(dataX, self.optimLogit)
        return ret
        
    def predictProba(self, dataX):
        ret = scipy.special.expit(self.predict(dataX))
        return ret
        
 
if __name__ == '__main__':
    data = sklearn.datasets.load_iris()
    x = data.data
    y = data.target

    for i in range(x.shape[0]):
        if y[i] != 0:
            y[i] = 1
    l = LogisticRegression(x, y)
    l.fit()
    out = l.predict(x)

    l2 = sklearn.linear_model.LogisticRegression()
    l2.fit(x, y)
    out2 = l2.predict_proba(x)

    pass


    
    






