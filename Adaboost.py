# -*- coding: utf-8 -*-

'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''

from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def weakClassifier(dataMatrix,dimen,threshVal,threshIneq):#just classify the data,,lt:less than
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def BuildweakClassifier(dataArr,classLabels,weight_):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = weakClassifier(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = weight_.T*errArr  #calc total error multiplied by weight_
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    weight_ = mat(ones((m,1))/m)   #init weight_ to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = BuildweakClassifier(dataArr,classLabels,weight_)#build Stump
        print "weight_:",weight_.T
        beta = float(0.5*log((1.0-error)/max(error,1e-16)))#calc beta, throw in max(error,eps) to account for error=0
        bestStump['beta'] = beta  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*beta*mat(classLabels).T,classEst) #exponent for weight_ calc, getting messy
        weight_ = multiply(weight_,exp(expon))                              #Calc New weight_ for next iteration
        weight_ = weight_/weight_.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += beta*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break
    print "#######################################################"    
    print "Number of Iteration: ",i+1
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = weakClassifier(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['beta']*classEst
    #print aggClassEst
    return sign(aggClassEst)


def change_data(x):
    x[0:100,0:50]=1;
    x[0:100,50:] = -1;
    return x


if __name__=="__main__":
    #dat,lab = loadSimpData()
    
    from sklearn import svm, datasets

    iris = datasets.load_iris();
    print 'type of iris: ', type(iris) #<class 'sklearn.datasets.base.Bunch'>
    print 'keys:', iris.keys() #['target_names', 'data', 'target', 'DESCR', 'feature_names']

    dat = iris.data[0:100,0:3] #only use the first two features        
    # change data
    dat = change_data(dat)
    
    lab = iris.target[0:100]
    lab[lab==0]=-1    

    weakClassArr,aggClassEst = adaBoostTrainDS(dat,lab)
    print weakClassArr
    pred = adaClassify(dat,weakClassArr)
    