---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="cuB0025o7Y3p" -->
# Off-Policy Evaluation for Slate Recommendation
<!-- #endregion -->

<!-- #region id="1FiasDgkd1TI" -->
## Imports
<!-- #endregion -->

```python id="MEaMMSa4ekZ7"
import numpy
import decimal
import scipy.sparse
import os
import sys
import os.path
import sklearn.model_selection
import sklearn.tree
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
from sklearn.externals import joblib
import scipy.linalg
import itertools
```

<!-- #region id="5VWgwAETe9uG" -->
## Settings
<!-- #endregion -->

```python id="7yXbiVETfD2E"
DATA_DIR="/content"
```

```python id="8eaiJL-ke_pb"
def get_feature_sets(datasetName):
    anchorURL = [0]
    bodyDoc = [0]
    
    if datasetName.startswith('MSLR'):
        for i in range(25):
            anchorURL.extend([5*i+2, 5*i+4])
            bodyDoc.extend([5*i+1, 5*i+3, 5*i+5])
        anchorURL.extend([126, 127, 128, 129, 130, 131])
        bodyDoc.extend([132,133])
        
    elif datasetName.startswith('MQ200'):
        for i in range(8):
            anchorURL.extend([5*i+2, 5*i+4])
            bodyDoc.extend([5*i+1, 5*i+3, 5*i+5])
        anchorURL.extend([41, 42, 43, 44, 45, 46])

    else:
        print("Settings:get_feature_sets [ERR] Unknown dataset. Use MSLR/MQ200*", flush=True)
        sys.exit(0)
        
    return anchorURL, bodyDoc
```

<!-- #region id="g3iPvTWlehc_" -->
## Datasets
<!-- #endregion -->

<!-- #region id="iz2zRXdTeihi" -->
Classes that pre-process datasets for semi-synthetic experiments
<!-- #endregion -->

```python id="CqXEXVlOeneC"
class Datasets:
    def __init__(self):
        #Must call either loadTxt(...)/loadNpz(...) to set all these members
        #before using Datasets objects elsewhere
        self.relevances=None
        self.features=None
        self.docsPerQuery=None
        self.queryMappings=None
        self.name=None
    
        #For filtered datasets, some docsPerQuery may be masked
        self.mask=None
        
    ###As a side-effect, loadTxt(...) stores a npz file for
    ###faster subsequent loading via loadNpz(...)
    #file_name: (str) Path to dataset file (.txt format)
    #name:      (str) String to identify this Datasets object henceforth
    def loadTxt(self, file_name, name):
        #Internal: Counters to keep track of docID and qID
        previousQueryID=None
        docID=None
        qID=0
        relevanceArray=None
        
        #QueryMappings: list[int],length=numQueries
        self.queryMappings=[]
        
        self.name=name
        
        #DocsPerQuery:  list[int],length=numQueries
        self.docsPerQuery=[]
        
        #Relevances:    list[Alpha],length=numQueries; Alpha:= numpy.array[int],length=docsForQuery
        self.relevances=[]
        
        #Features:      list[Alpha],length=numQueries; 
        #Alpha:= scipy.sparse.coo_matrix[double],shape=(docsForQuery, numFeatures)
        featureRows=None
        featureCols=None
        featureVals=None
        
        self.features=[]
        numFeatures=None
        
        #Now read in data
        with open(file_name, 'r') as f:
            outputFilename=file_name[:-4]
            outputFileDir=outputFilename+'_processed'
            if not os.path.exists(outputFileDir):
                os.makedirs(outputFileDir)
            
            for line in f:
                tokens=line.split(' ', 2)
                relevance=int(tokens[0])
                queryID=int(tokens[1].split(':', 1)[1])
                
                #Remove any trailing comments before extracting features
                remainder=tokens[2].split('#', 1)
                featureTokens=remainder[0].strip().split(' ')
                
                if numFeatures is None:
                    numFeatures=len(featureTokens)+1
                    
                if (previousQueryID is None) or (queryID!=previousQueryID):
                    #Begin processing a new query's documents
                    docID=0
                    
                    if relevanceArray is not None:
                        #Previous query's data should be persisted to file/self.members
                        currentRelevances=numpy.array(relevanceArray, 
                                                dtype=numpy.int, copy=False)
                        self.relevances.append(currentRelevances)
                        numpy.savez_compressed(os.path.join(outputFileDir, str(qID)+'_rel'), 
                                                relevances=currentRelevances)
                        
                        maxDocs=len(relevanceArray)
                        self.docsPerQuery.append(maxDocs)
                        
                        currentFeatures=scipy.sparse.coo_matrix((featureVals, (featureRows, featureCols)),
                                                shape=(maxDocs, numFeatures), dtype=numpy.float64)
                        currentFeatures=currentFeatures.tocsr()
                        self.features.append(currentFeatures)
                        scipy.sparse.save_npz(os.path.join(outputFileDir, str(qID)+'_feat'), 
                                                currentFeatures)
        
                        qID+=1
                        self.queryMappings.append(previousQueryID)
                        
                        if len(self.docsPerQuery)%100==0:
                            print(".", end="", flush=True)
                            
                    relevanceArray=[]
                    featureRows=[]
                    featureCols=[]
                    featureVals=[]
                    
                    previousQueryID=queryID
                else:
                    docID+=1
                    
                relevanceArray.append(relevance)
                
                #Add a feature for the the intercept
                featureRows.append(docID)
                featureCols.append(0)
                featureVals.append(0.01)
                
                for featureToken in featureTokens:
                    featureTokenSplit=featureToken.split(':', 1)
                    featureIndex=int(featureTokenSplit[0])
                    featureValue=float(featureTokenSplit[1])
                    
                    featureRows.append(docID)
                    featureCols.append(featureIndex)
                    featureVals.append(featureValue)
            
            #Finish processing the final query's data
            currentRelevances=numpy.array(relevanceArray, dtype=numpy.int, copy=False)
            self.relevances.append(currentRelevances)
            numpy.savez_compressed(os.path.join(outputFileDir, str(qID)+'_rel'), 
                                        relevances=currentRelevances)
            
            maxDocs=len(relevanceArray)
            self.docsPerQuery.append(maxDocs)
            
            currentFeatures=scipy.sparse.coo_matrix((featureVals, (featureRows, featureCols)),
                                        shape=(maxDocs, numFeatures), dtype=numpy.float64)
            currentFeatures=currentFeatures.tocsr()
            self.features.append(currentFeatures)
            scipy.sparse.save_npz(os.path.join(outputFileDir, str(qID)+'_feat'),
                                        currentFeatures)
            
            self.queryMappings.append(previousQueryID)
        
        #Persist meta-data for the dataset for faster loading through loadNpz
        numpy.savez_compressed(outputFilename, docsPerQuery=self.docsPerQuery, 
                                        name=self.name, queryMappings=self.queryMappings)
        
        print("", flush=True)
        print("Datasets:loadTxt [INFO] Loaded", file_name, 
                    "\t NumQueries", len(self.docsPerQuery), 
                    "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery), 
                    max(self.docsPerQuery), flush=True)
    
    #file_name: (str) Path to dataset file/directory
    def loadNpz(self, file_name):
        with numpy.load(file_name+'.npz') as npFile:
            self.docsPerQuery=npFile['docsPerQuery']
            self.name=str(npFile['name'])
            self.queryMappings=npFile['queryMappings']
        
        fileDir = file_name+'_processed'
        if os.path.exists(fileDir):
            self.relevances=[]
            self.features=[]
            
            qID=0
            while os.path.exists(os.path.join(fileDir, str(qID)+'_rel.npz')):
                with numpy.load(os.path.join(fileDir, str(qID)+'_rel.npz')) as currRelFile:
                    self.relevances.append(currRelFile['relevances'])
                
                self.features.append(scipy.sparse.load_npz(os.path.join(fileDir, str(qID)+'_feat.npz')))
                    
                qID+=1
                
                if qID%100==0:
                    print(".", end="", flush=True)
                
        print("", flush=True)
        print("Datasets:loadNpz [INFO] Loaded", file_name, "\t NumQueries", len(self.docsPerQuery), 
                    "\t [Min/Max]DocsPerQuery", min(self.docsPerQuery), 
                    max(self.docsPerQuery), "\t [Sum] docsPerQuery", sum(self.docsPerQuery), flush=True)
```

```python id="6hXZf6YhexDv"
"""
mq2008Data=Datasets()
mq2008Data.loadTxt(Settings.DATA_DIR+'MQ2008.txt', 'MQ2008')
mq2008Data.loadNpz(Settings.DATA_DIR+'MQ2008')
del mq2008Data

mq2007Data=Datasets()
mq2007Data.loadTxt(Settings.DATA_DIR+'MQ2007.txt', 'MQ2007')
mq2007Data.loadNpz(Settings.DATA_DIR+'MQ2007')
del mq2007Data
"""
mslrData=Datasets()
mslrData.loadTxt(Settings.DATA_DIR+'MSLR-WEB10K/mslr.txt', 'MSLR10k')
del mslrData

for foldID in range(1,6):
    for fraction in ['train','vali','test']:
        mslrData=Datasets()
        mslrData.loadTxt(Settings.DATA_DIR+'MSLR-WEB10K\\Fold'+str(foldID)+'\\'+fraction+'.txt', 'MSLR10k-'+str(foldID)+'-'+fraction)
        del mslrData

mslrData=Datasets()
mslrData.loadTxt(Settings.DATA_DIR+'MSLR/mslr.txt', 'MSLR')
del mslrData

for foldID in range(1,6):
    for fraction in ['train','vali','test']:
        mslrData=Datasets()
        mslrData.loadTxt(Settings.DATA_DIR+'MSLR\\Fold'+str(foldID)+'\\'+fraction+'.txt', 'MSLR-'+str(foldID)+'-'+fraction)
        del mslrData
```

<!-- #region id="pgow506kfOuB" -->
## Estimators
<!-- #endregion -->

```python id="ELjwikRQfPdt"
class Estimator:
    #ranking_size: (int) Size of slate, l
    #logging_policy: (UniformPolicy) Logging policy, \mu
    #target_policy: (Policy) Target policy, \pi
    def __init__(self, ranking_size, logging_policy, target_policy):
        self.rankingSize=ranking_size
        self.name=None
        self.loggingPolicy=logging_policy
        self.targetPolicy=target_policy
        
        if target_policy.name is None or logging_policy.name is None:
            print("Estimator:init [ERR] Either target or logging policy is not initialized", flush=True)
            sys.exit(0)
            
        if target_policy.dataset.name != logging_policy.dataset.name:
            print("Estimator:init [ERR] Target and logging policy operate on different datasets", flush=True)
            sys.exit(0)
            
        ###All sub-classes of Estimator should supply a estimate method
        ###Requires: query, logged_ranking, logged_value,
        ###Returns: float indicating estimated value
        
        self.runningSum=0
        self.runningMean=0.0

    def updateRunningAverage(self, value):
        self.runningSum+=1
        delta=value-self.runningMean
        self.runningMean+=delta/self.runningSum

    def reset(self):
        self.runningSum=0
        self.runningMean=0.0 
```

```python id="bhcE-LH5gE4n"
 class OnPolicy(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy, metric):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='OnPolicy'
        self.metric=metric
        
        #This member is set on-demand by estimateAll(...)
        self.savedValues=None
        
    def estimateAll(self):
        if self.savedValues is not None:
            return
            
        self.savedValues=[]
        numQueries=len(self.loggingPolicy.dataset.docsPerQuery)
        for i in range(numQueries):
            newRanking=self.targetPolicy.predict(i, self.rankingSize)
            self.savedValues.append(self.metric.computeMetric(i, newRanking))
            if i%100==0:
                print(".", end="", flush=True)
                
        print("")
        print("OnPolicy:estimateAll [LOG] Precomputed estimates.", flush=True)
            
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        currentValue=None
        if self.savedValues is not None:
            currentValue=self.savedValues[query]
        else:
            currentValue=self.metric.computeMetric(query, new_ranking)
            
        self.updateRunningAverage(currentValue)
        return self.runningMean
        
    def reset(self):
        Estimator.reset(self)
        self.savedValues=None
```

<!-- #region id="3PVjYN3Sf_d4" -->
### Uniform IPS
<!-- #endregion -->

```python id="JhQTQ6DDf--k"
class UniformIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='Unif-IPS'
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch=numpy.absolute(new_ranking-logged_ranking).sum() == 0
        currentValue=0.0
        if exactMatch:
            numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
            validDocs=logged_ranking.size
            invPropensity=None
            if self.loggingPolicy.allowRepetitions:
                invPropensity=numpy.float_power(numAllowedDocs, validDocs)
            else:
                invPropensity=numpy.prod(range(numAllowedDocs+1-validDocs, numAllowedDocs+1), dtype=numpy.float64)
                
            currentValue=logged_value*invPropensity

        self.updateRunningAverage(currentValue)
        return self.runningMean
```

<!-- #region id="hq7Mn459f7e1" -->
### Non-uniform IPS
<!-- #endregion -->

```python id="H-3GhjpEf62b"
class NonUniformIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='NonUnif-IPS'
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch=numpy.absolute(new_ranking-logged_ranking).sum() == 0
        currentValue=0.0
        if exactMatch:
            numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
            underlyingRanking=self.loggingPolicy.policy.predict(query, -1)
            currentDistribution=self.loggingPolicy.multinomials[numAllowedDocs]
            
            numRankedDocs=logged_ranking.size
            invPropensity=1.0
            denominator=1.0
            for j in range(numRankedDocs):
                underlyingIndex=numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                invPropensity*=(denominator*1.0/currentDistribution[underlyingIndex])
                if not self.loggingPolicy.allowRepetitions:
                    denominator-=currentDistribution[underlyingIndex]
                
            currentValue=logged_value*invPropensity

        self.updateRunningAverage(currentValue)
        return self.runningMean
```

<!-- #region id="WP3SANInf3LK" -->
### Uniform SNIPS
<!-- #endregion -->

```python id="__7sXcHAf2jU"
class UniformSNIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='Unif-IPS_SN'
        self.runningDenominatorMean=0.0
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch=numpy.absolute(new_ranking-logged_ranking).sum() == 0
        currentValue=0.0
        if exactMatch:
            numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
            validDocs=logged_ranking.size
            invPropensity=None
            if self.loggingPolicy.allowRepetitions:
                invPropensity=numpy.float_power(numAllowedDocs, validDocs)
            else:
                invPropensity=numpy.prod(range(numAllowedDocs+1-validDocs, numAllowedDocs+1), dtype=numpy.float64)
                
            currentValue=logged_value*invPropensity

            self.updateRunningAverage(currentValue)
            denominatorDelta=invPropensity-self.runningDenominatorMean
            self.runningDenominatorMean+=denominatorDelta/self.runningSum
        if self.runningDenominatorMean!=0.0:
            return 1.0*self.runningMean/self.runningDenominatorMean
        else:
            return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean=0.0
```

<!-- #region id="J5_uSseWfyGL" -->
### Non-uniform SNIPS
<!-- #endregion -->

```python id="Qb3AsxMqfxcG"
class NonUniformSNIPS(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='NonUnif-IPS_SN'
        self.runningDenominatorMean=0.0
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        exactMatch=numpy.absolute(new_ranking-logged_ranking).sum() == 0
        currentValue=0.0
        if exactMatch:
            numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
            underlyingRanking=self.loggingPolicy.policy.predict(query, -1)
            currentDistribution=self.loggingPolicy.multinomials[numAllowedDocs]
            
            numRankedDocs=logged_ranking.size
            invPropensity=1.0
            denominator=1.0
            for j in range(numRankedDocs):
                underlyingIndex=numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
                invPropensity*=(denominator*1.0/currentDistribution[underlyingIndex])
                if not self.loggingPolicy.allowRepetitions:
                    denominator-=currentDistribution[underlyingIndex]
                
            currentValue=logged_value*invPropensity

            self.updateRunningAverage(currentValue)
            denominatorDelta=invPropensity-self.runningDenominatorMean
            self.runningDenominatorMean+=denominatorDelta/self.runningSum
        if self.runningDenominatorMean!=0.0:
            return 1.0*self.runningMean/self.runningDenominatorMean
        else:
            return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean=0.0
```

<!-- #region id="okkeE_aGftlY" -->
### Uniform PI
<!-- #endregion -->

```python id="dWwKVEPdftBv"
class UniformPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='Unif-PI'
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
        validDocs=logged_ranking.size
        vectorDimension=validDocs*numAllowedDocs
        
        exploredMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            if self.loggingPolicy.dataset.mask is None:
                exploredMatrix[j, logged_ranking[j]]=1
                newMatrix[j, new_ranking[j]]=1
            else:
                logIndex=numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == logged_ranking[j])[0]
                newIndex=numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == new_ranking[j])[0]
                exploredMatrix[j, logIndex]=1
                newMatrix[j, newIndex]=1
        
        posRelVector=exploredMatrix.reshape(vectorDimension)
        newSlateVector=newMatrix.reshape(vectorDimension)
        
        estimatedPhi=numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity=numpy.dot(estimatedPhi, newSlateVector)
        
        currentValue=logged_value*invPropensity
        
        self.updateRunningAverage(currentValue)
        return self.runningMean
```

<!-- #region id="G_bvQ13mfoY5" -->
### Non-uniform PI
<!-- #endregion -->

```python id="Sb6AzI-Vfn2d"
class NonUniformPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='NonUnif-PI'
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
        underlyingRanking=self.loggingPolicy.policy.predict(query, -1)
        validDocs=logged_ranking.size
        vectorDimension=validDocs*numAllowedDocs
        
        exploredMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            logIndex=numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
            newIndex=numpy.flatnonzero(underlyingRanking == new_ranking[j])[0]
            exploredMatrix[j, logIndex]=1
            newMatrix[j, newIndex]=1
        
        posRelVector=exploredMatrix.reshape(vectorDimension)
        newSlateVector=newMatrix.reshape(vectorDimension)
        
        estimatedPhi=numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity=numpy.dot(estimatedPhi, newSlateVector)
 
        currentValue=logged_value*invPropensity
        
        self.updateRunningAverage(currentValue)
        return self.runningMean
```

<!-- #region id="rNfsM1yjfkdd" -->
### Uniform SNPI
<!-- #endregion -->

```python id="lv2FX7dhfg5L"
class UniformSNPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='Unif-PI_SN'
        self.runningDenominatorMean=0.0

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
        validDocs=logged_ranking.size
        vectorDimension=validDocs*numAllowedDocs
        
        exploredMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            if self.loggingPolicy.dataset.mask is None:
                exploredMatrix[j, logged_ranking[j]]=1
                newMatrix[j, new_ranking[j]]=1
            else:
                logIndex=numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == logged_ranking[j])[0]
                newIndex=numpy.flatnonzero(self.loggingPolicy.dataset.mask[query] == new_ranking[j])[0]
                exploredMatrix[j, logIndex]=1
                newMatrix[j, newIndex]=1
        
        posRelVector=exploredMatrix.reshape(vectorDimension)
        newSlateVector=newMatrix.reshape(vectorDimension)
        
        estimatedPhi=numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity=numpy.dot(estimatedPhi, newSlateVector)
        currentValue=logged_value*invPropensity
        
        self.updateRunningAverage(currentValue)
        
        denominatorDelta=invPropensity-self.runningDenominatorMean
        self.runningDenominatorMean+=denominatorDelta/self.runningSum
        if self.runningDenominatorMean!=0.0:
            return 1.0*self.runningMean/self.runningDenominatorMean
        else:
            return 0.0

    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean=0.0
```

<!-- #region id="qpheFMdMfcXN" -->
### Non-uniform SNPI
<!-- #endregion -->

```python id="gIr66upPfbYq"
class NonUniformSNPI(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name='NonUnif-PI_SN'
        self.runningDenominatorMean=0.0

    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        numAllowedDocs=self.loggingPolicy.dataset.docsPerQuery[query]
        underlyingRanking=self.loggingPolicy.policy.predict(query, -1)
        
        validDocs=logged_ranking.size
        vectorDimension=validDocs*numAllowedDocs
        
        exploredMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        newMatrix=numpy.zeros((validDocs, numAllowedDocs), dtype=numpy.float64)
        for j in range(validDocs):
            logIndex=numpy.flatnonzero(underlyingRanking == logged_ranking[j])[0]
            newIndex=numpy.flatnonzero(underlyingRanking == new_ranking[j])[0]
            exploredMatrix[j, logIndex]=1
            newMatrix[j, newIndex]=1
        
        posRelVector=exploredMatrix.reshape(vectorDimension)
        newSlateVector=newMatrix.reshape(vectorDimension)
        
        estimatedPhi=numpy.dot(self.loggingPolicy.gammas[numAllowedDocs], posRelVector)
        invPropensity=numpy.dot(estimatedPhi, newSlateVector)
        currentValue=logged_value*invPropensity
        
        self.updateRunningAverage(currentValue)
        
        denominatorDelta=invPropensity-self.runningDenominatorMean
        self.runningDenominatorMean+=denominatorDelta/self.runningSum
        if self.runningDenominatorMean!=0.0:
            return 1.0*self.runningMean/self.runningDenominatorMean
        else:
            return 0.0
    
    def reset(self):
        Estimator.reset(self)
        self.runningDenominatorMean=0.0
```

<!-- #region id="xEQSGR1jfXvQ" -->
### DM
<!-- #endregion -->

```python id="YHrGViwrfWdM"
class Direct(Estimator):
    def __init__(self, ranking_size, logging_policy, target_policy, estimator_type):
        Estimator.__init__(self, ranking_size, logging_policy, target_policy)
        self.name = 'Direct_'+estimator_type
        self.estimatorType = estimator_type
        self.numFeatures=self.loggingPolicy.dataset.features[0].shape[1]
        self.hyperParams={'alpha': (numpy.logspace(-2,1,num=4,base=10)).tolist()}
        self.treeDepths={'max_depth': list(range(3,15,3))}
        
        if self.estimatorType=='tree':
            self.tree=None
        else:
            self.policyParams=None
            
        #This member is set on-demand by estimateAll(...)
        self.savedValues=None
        
    def estimateAll(self, metric=None):
        if self.savedValues is not None:
            return
            
        self.savedValues=[]
        numQueries=len(self.loggingPolicy.dataset.docsPerQuery)
        for query in range(numQueries):
            newRanking=self.targetPolicy.predict(query, self.rankingSize)
            allFeatures=self.loggingPolicy.dataset.features[query][newRanking,:]
        
            if newRanking.size < self.rankingSize:
                emptyPad=scipy.sparse.csr_matrix((self.rankingSize-newRanking.size, self.numFeatures), dtype=numpy.float64)
                allFeatures=scipy.sparse.vstack((allFeatures, emptyPad), format="csr", dtype=numpy.float64)
            
            allFeatures=allFeatures.toarray()
            nRows, nCols = allFeatures.shape
            size=nRows*nCols
            currentFeatures=numpy.reshape(allFeatures, (1,size))
        
            currentValue=None
            if self.estimatorType=='tree':
                currentValue=self.tree.predict(currentFeatures)[0]
            else:
                currentValue=numpy.dot(currentFeatures, self.policyParams)[0]
            
            low=None
            high=None
            if metric is not None:
                low=metric.getMin(newRanking.size)
                high=metric.getMax(newRanking.size)
                
            if low is not None:
                currentValue = max(currentValue, low)
            if high is not None:
                currentValue = min(currentValue, high)

            if currentValue > 1.0 or currentValue < 0.0:
                print("Direct:estimateAll [LOG] estimate %0.3f " % (currentValue), flush=True)

            del allFeatures
            del currentFeatures
            
            self.savedValues.append(currentValue)
            
            if query%100==0:
                print(".", end="", flush=True)
                
        print("")
        print("Direct:estimateAll [LOG] Precomputed estimates.", flush=True)
        
    def train(self, logged_data):
        numInstances=len(logged_data)
        targets=numpy.zeros(numInstances, order='C', dtype=numpy.float64)
        covariates=scipy.sparse.lil_matrix((numInstances, self.numFeatures*self.rankingSize), dtype=numpy.float64)
        print("Starting to create covariates", flush=True)
        for j in range(numInstances):
            currentDatapoint=logged_data.pop()
            targets[j]=currentDatapoint[2]
            
            currentQuery=currentDatapoint[0]
            currentRanking=currentDatapoint[1]
            allFeatures=self.loggingPolicy.dataset.features[currentQuery][currentRanking,:]
            allFeatures.eliminate_zeros()
            
            covariates.data[j]=allFeatures.data
            newIndices=allFeatures.indices
            for k in range(allFeatures.shape[0]):
                newIndices[allFeatures.indptr[k]:allFeatures.indptr[k+1]]+=k*self.numFeatures
                
            covariates.rows[j]=newIndices
                
            if j%1000 == 0:
                print(".", end='', flush=True)
            del currentDatapoint
            del allFeatures

            
        print("Converting covariates", flush=True)
        covariates=covariates.tocsr()
        print("Finished conversion", flush=True)
        
        if self.estimatorType=='tree':
            treeCV=sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeRegressor(criterion="mse",
                                                        splitter="random", min_samples_split=4, 
                                                        min_samples_leaf=4, presort=False),
                                param_grid=self.treeDepths,
                                scoring=None, fit_params=None, n_jobs=1,
                                iid=True, cv=3, refit=True, verbose=0, pre_dispatch=1,
                                error_score='raise', return_train_score=False)
            treeCV.fit(covariates, targets)
            self.tree=treeCV.best_estimator_
            print("DirectEstimator:train [INFO] Done. Best depth", 
                            treeCV.best_params_['max_depth'], flush=True)
        elif self.estimatorType=='lasso':
            lassoCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(fit_intercept=False, 
                                                        normalize=False, precompute=False, copy_X=False, 
                                                        max_iter=30000, tol=1e-4, warm_start=False, positive=False,
                                                        random_state=None, selection='random'),
                                param_grid=self.hyperParams,
                                scoring=None, fit_params=None, n_jobs=1,
                                iid=True, cv=3, refit=True, verbose=0, pre_dispatch=1,
                                error_score='raise', return_train_score=False)
            lassoCV.fit(covariates, targets)
            self.policyParams=lassoCV.best_estimator_.coef_
            print("DirectEstimator:train [INFO] Done. CVAlpha", lassoCV.best_params_['alpha'], flush=True)
        elif self.estimatorType=='ridge':
            ridgeCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(fit_intercept=False,
                                                        normalize=False, copy_X=False, max_iter=30000, tol=1e-4, solver='sag',
                                                        random_state=None),
                                param_grid=self.hyperParams,
                                scoring=None, fit_params=None, n_jobs=1,
                                iid=True, cv=3, refit=True, verbose=0, pre_dispatch=1,
                                error_score='raise', return_train_score=False)
            ridgeCV.fit(covariates, targets)
            self.policyParams=ridgeCV.best_estimator_.coef_
            print("DirectEstimator:train [INFO] Done. CVAlpha", ridgeCV.best_params_['alpha'], flush=True)
        else:
            print("DirectEstimator:train [ERR] %s not supported." % self.modelType, flush=True)
            sys.exit(0)
        
    def estimate(self, query, logged_ranking, new_ranking, logged_value):
        currentValue=None
        if self.savedValues is not None:
            currentValue=self.savedValues[query]
        else:
            allFeatures=self.loggingPolicy.dataset.features[query][new_ranking,:]
        
            if new_ranking.size < self.rankingSize:
                emptyPad=scipy.sparse.csr_matrix((self.rankingSize-new_ranking.size, self.numFeatures), dtype=numpy.float64)
                allFeatures=scipy.sparse.vstack((allFeatures, emptyPad), format="csr", dtype=numpy.float64)
            
            allFeatures=allFeatures.toarray()
            nRows, nCols = allFeatures.shape
            size=nRows*nCols
            currentFeatures=numpy.reshape(allFeatures, (1,size))

            if self.estimatorType=='tree':
                currentValue=self.tree.predict(currentFeatures)[0]
            else:
                currentValue=numpy.dot(currentFeatures, self.policyParams)[0]
        
            del allFeatures
            del currentFeatures
            
        self.updateRunningAverage(currentValue)
        return self.runningMean
        
    def reset(self):
        Estimator.reset(self)
        self.savedValues=None
        if self.estimatorType=='tree':
            self.tree=None
        else:
            self.policyParams=None
```

<!-- #region id="zSm9QoicgUax" -->
## Utils
<!-- #endregion -->

```python id="HSSuW74bgWA9"
class GammaCalculator:
    
    def __init__(self, weights, nSlots):
        assert nSlots>0, "NSLOTS MUST BE POSITIVE"
        self.nDocs = len(weights)
        self.nSlots = nSlots

        self.nTypes = 0
        self.weightToType = {}
        self.typeToWeight = []
        self.typeToDocs = []
        self.nDocsOfType = []
        self.docToType = []

        self.weights = weights

        for i in range(len(weights)):
            weight = weights[i]
            if not weight in self.weightToType:
                self.typeToWeight.append(decimal.Decimal(weight))
                self.typeToDocs.append([])
                self.nDocsOfType.append(0)
                self.weightToType[weight] = self.nTypes
                self.nTypes += 1

            t = self.weightToType[weight]
            self.docToType.append(t)
            self.nDocsOfType[t] += 1
            self.typeToDocs[t].append(i)

        self.table = {}
        empty_prefix = (0,)*self.nTypes
        self.table[ empty_prefix, () ] = decimal.Decimal(1)
        self.visited = set()
        self.fill_table(empty_prefix, ())

        self.gamma_types = {}
        for (prefix,anchor) in self.table.keys():
            length = sum(prefix)
            for t in range(self.nTypes):
                if prefix[t]<self.nDocsOfType[t]:
                    prob = self.get_prob(prefix, anchor, t)
                    if anchor==():
                        key = "types1", (length,t)
                    else:
                        key = "types2", anchor, (length,t)
                    if not key in self.gamma_types:
                        self.gamma_types[key] = decimal.Decimal(0)
                    self.gamma_types[key] += prob

        self.unitMarginals = numpy.zeros((self.nSlots, self.nDocs), dtype = numpy.longdouble)
        self.pairwiseMarginals = {}
        for (key, prob) in self.gamma_types.items():
            if key[0]=="types1":
                pos, t = key[1]
                normalize = decimal.Decimal(self.nDocsOfType[t])
                for d in self.typeToDocs[t]:
                    self.unitMarginals[pos, d] = numpy.longdouble(prob/normalize)

            if key[0]=="types2":
                pos1, t1 = key[1]
                pos2, t2 = key[2]
                normalize = None
                if t1==t2:
                    normalize = decimal.Decimal(self.nDocsOfType[t1]*(self.nDocsOfType[t2]-1))
                else:
                    normalize = decimal.Decimal(self.nDocsOfType[t1]*self.nDocsOfType[t2])

                newKey = (pos1, pos2)
                if newKey not in self.pairwiseMarginals:
                    self.pairwiseMarginals[newKey] = numpy.zeros((self.nDocs, self.nDocs), dtype = numpy.longdouble)
         
                for d1 in self.typeToDocs[t1]:
                    for d2 in self.typeToDocs[t2]:
                        if d1 != d2:
                            self.pairwiseMarginals[newKey][d1, d2] = numpy.longdouble(prob/normalize)
            
    def decr(self, prefix, t):
        prefix_mut = list(prefix)
        assert prefix_mut[t]>0, "DECR PREFIX OUT OF BOUNDS"
        prefix_mut[t] -= 1
        return tuple(prefix_mut)

    def incr(self, prefix, t):
        prefix_mut = list(prefix)
        assert prefix_mut[t]<self.nDocsOfType[t], "INCR PREFIX OUT OF BOUNDS"
        prefix_mut[t] += 1
        return tuple(prefix_mut)

    def get_prob(self, prefix, anchor, t):
        posterior = [ self.typeToWeight[tt]*(self.nDocsOfType[tt]-prefix[tt]) for tt in range(self.nTypes) ]
        normalize = sum(posterior)
        return self.eval_table(prefix, anchor) * posterior[t] / normalize

    def eval_table(self, prefix, anchor):
        """evaluate an entry in the DP table. here:
              prefix: tuple of type counts
              anchor: specifies (pos, type) where pos<len(prefix)"""

        if (prefix,anchor) in self.table:
            return self.table[prefix,anchor]

        prob = decimal.Decimal(0)
        length = sum(prefix)
        if anchor==() or anchor[0]<length-1:
            for t in range(self.nTypes):
                if prefix[t]>0:
                    prefix0 = self.decr(prefix, t)
                    if anchor==() or prefix0[anchor[1]]>0:
                        prob += self.get_prob(prefix0, anchor, t)
        else:
            t=anchor[1]
            prefix0 = self.decr(prefix, t)
            prob += self.get_prob(prefix0, (), t)
        self.table[prefix,anchor] = prob
        return prob
        
    def fill_table(self, prefix, anchor):
        """add more entries to the DP table extending the current prefix. here:
              prefix: tuple of type counts
              anchor: specifies (pos, type) where pos<len(prefix)"""

        length = sum(prefix)
        if (prefix, anchor) in self.visited:
            return
        self.visited.add( (prefix, anchor) )
        self.eval_table(prefix, anchor)
        if length==self.nSlots-1:
            return

        for t in range(self.nTypes):
            if prefix[t]<self.nDocsOfType[t]:
                prefix1 = self.incr(prefix, t)
                anchor1 = (length, t)
                self.fill_table(prefix1, anchor)
                if anchor==():
                    self.fill_table(prefix1, anchor1)
```

<!-- #region id="vgBuPEK0gfmh" -->
## Metrics
<!-- #endregion -->

```python id="ZbLiCBPsgfkh"
class Metric:
    #dataset: (Datasets) Must be initialized using Datasets.loadTxt(...)/loadNpz(...)
    #ranking_size: (int) Maximum size of slate across contexts, l
    def __init__(self, dataset, ranking_size):
        self.rankingSize=ranking_size
        self.dataset=dataset
        self.name=None
        ###All sub-classes of Metric should supply a computeMetric method
        ###Requires: (int) query_id; list[int],length=ranking_size ranking
        ###Returns: (double) value


class ConstantMetric(Metric):
    #constant: (double) Value returned by this metric
    def __init__(self, dataset, ranking_size, constant):
        Metric.__init__(self, dataset, ranking_size)
        self.constant=constant
        self.name='Constant'
        print("ConstantMetric:init [INFO] RankingSize", ranking_size, "\t Constant", constant, flush=True)
    
    #query_id: (int) Index of the query                                                                 (unused)
    #ranking: list[int],length=min(ranking_size,docsForQuery); Valid DocID in each slot of the slate    (unused)
    def computeMetric(self, query_id, ranking):
        return self.constant

        
class DCG(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.discountParams=1.0+numpy.array(range(self.rankingSize), dtype=numpy.float64)
        self.discountParams[0]=2.0
        self.discountParams[1]=2.0
        self.discountParams=numpy.reciprocal(numpy.log2(self.discountParams))
        self.name='DCG'
        print("DCG:init [INFO] RankingSize", ranking_size, flush=True)
    
    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        gain=numpy.exp2(relevanceList)-1.0
        dcg=numpy.dot(self.discountParams[0:numpy.shape(gain)[0]], gain)
        return dcg

        
class NDCG(Metric):
    #allow_repetitions: (bool) If True, max gain is computed as if repetitions are allowed in the ranking
    def __init__(self, dataset, ranking_size, allow_repetitions):
        Metric.__init__(self, dataset, ranking_size)
        self.discountParams=1.0+numpy.array(range(self.rankingSize), dtype=numpy.float64)
        self.discountParams[0]=2.0
        self.discountParams[1]=2.0
        self.discountParams=numpy.reciprocal(numpy.log2(self.discountParams))
        self.name='NDCG'
        
        self.normalizers=[]
        numQueries=len(self.dataset.docsPerQuery)
        for currentQuery in range(numQueries):
            validDocs=min(self.dataset.docsPerQuery[currentQuery], ranking_size)
            currentRelevances=self.dataset.relevances[currentQuery]
            
            #Handle filtered datasets properly
            if self.dataset.mask is not None:
                currentRelevances=currentRelevances[self.dataset.mask[currentQuery]]
            
            maxRelevances=None
            if allow_repetitions:
                maxRelevances=numpy.repeat(currentRelevances.max(), validDocs)
            else:
                maxRelevances=-numpy.sort(-currentRelevances)[0:validDocs]
        
            maxGain=numpy.exp2(maxRelevances)-1.0
            maxDCG=numpy.dot(self.discountParams[0:validDocs], maxGain)
            
            self.normalizers.append(maxDCG)
            
            if currentQuery % 1000==0:
                print(".", end="", flush=True)
                
        print("", flush=True)        
        print("NDCG:init [INFO] RankingSize", ranking_size, "\t AllowRepetitions?", allow_repetitions, flush=True)
    
    def computeMetric(self, query_id, ranking):
        normalizer=self.normalizers[query_id]
        if normalizer<=0.0:
            return 0.0
        else:
            relevanceList=self.dataset.relevances[query_id][ranking]
            gain=numpy.exp2(relevanceList)-1.0
            dcg=numpy.dot(self.discountParams[0:numpy.shape(gain)[0]], gain)
            return dcg*1.0/normalizer

    def getMax(self,ranking_size):
        return 1.0
            
    def getMin(self,ranking_size):
        return 0.0
   
class ERR(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='ERR'
        
        #ERR needs the maximum relevance grade for the dataset
        #For MQ200*, this is 2;  For MSLR, this is 4
        self.maxrel=None
        if self.dataset.name.startswith('MSLR'):
            self.maxrel=numpy.exp2(4)
        elif self.dataset.name.startswith('MQ200'):
            self.maxrel=numpy.exp2(2)
        else:
            print("ERR:init [ERR] Unknown dataset. Use MSLR/MQ200*", flush=True)
            sys.exit(0)
        
        print("ERR:init [INFO] RankingSize", ranking_size, flush=True)

    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        gain=numpy.exp2(relevanceList)-1.0
        probs=gain*1.0/self.maxrel
        validDocs=numpy.shape(probs)[0]
        err=0.0
        p=1.0
        for i in range(validDocs):
            err+=p*probs[i]/(i+1)
            p=p*(1-probs[i])
        return err

    def getMax(self, ranking_size):
        probs=[(self.maxrel-1.0)/self.maxrel for i in range(ranking_size)]
        validDocs=numpy.shape(probs)[0]
        err=0.0
        p=1.0
        for i in range(validDocs):
            err+=p*probs[i]/(i+1)
            p=p*(1-probs[i])
        return err

    def getMin(self, ranking_size):
        return 0.0
        
class MaxRelevance(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='MaxRelevance'
        print("MaxRelevance:init [INFO] RankingSize", ranking_size, flush=True)
        
    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        maxRelevance=1.0*relevanceList.max()
        return maxRelevance

        
class SumRelevance(Metric):
    def __init__(self, dataset, ranking_size):
        Metric.__init__(self, dataset, ranking_size)
        self.name='SumRelevance'
        print("SumRelevance:init [INFO] RankingSize", ranking_size, flush=True)
        
    def computeMetric(self, query_id, ranking):
        relevanceList=self.dataset.relevances[query_id][ranking]
        sumRelevance=relevanceList.sum(dtype=numpy.float64)
        return sumRelevance

        
        
if __name__=="__main__":
    import Settings
    import Datasets
    
    mslrData = Datasets.Datasets()
    mslrData.loadNpz(Settings.DATA_DIR+"mslr/mslr")
    
    const=ConstantMetric(mslrData, 4, 5.0)
    print("Constant", const.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del const
    
    dcg=DCG(mslrData, 4)
    print("DCG", dcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del dcg
    
    ndcg=NDCG(mslrData, 4, False)
    print("NDCG NoRep", ndcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del ndcg
    
    ndcg=NDCG(mslrData, 4, True)
    print("NDCG YesRep", ndcg.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del ndcg
    
    err=ERR(mslrData, 4)
    print("ERR", err.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del err
    
    maxrel=MaxRelevance(mslrData, 4)
    print("MaxRelevance", maxrel.computeMetric(0, [0, 1, 2, 3]), flush=True)
    del maxrel
    
    sumrel=SumRelevance(mslrData, 4)
    print("SumRelevance", sumrel.computeMetric(0, [0, 1, 2, 3]), flush=True)  
    del sumrel
```

<!-- #region id="GZVo3XE9gfia" -->
## Policy
<!-- #endregion -->

<!-- #region id="aeOnCXqogffX" -->
Class that models a policy for exploration or evaluation
<!-- #endregion -->

```python id="xEXrtKIJgpF6"
#UniformGamma(...) computes a Gamma_pinv matrix for uniform exploration
#num_candidates: (int) Number of candidates, m
#ranking_size: (int) Size of slate, l
#allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def UniformGamma(num_candidates, ranking_size, allow_repetitions):
    validDocs=ranking_size
    if not allow_repetitions:
        validDocs=min(ranking_size, num_candidates)
                
    gamma=numpy.empty((num_candidates*validDocs, num_candidates*validDocs), dtype=numpy.float64)
    if num_candidates==1:
        gamma.fill(1.0)
    else:
        #First set all the off-diagonal blocks
        if allow_repetitions:
            gamma.fill(1.0/(num_candidates*num_candidates))
        else:
            gamma.fill(1.0/(num_candidates*(num_candidates-1)))
            #Correct the diagonal of each off-diagonal block: Pairwise=0
            for p in range(1,validDocs):
                diag=numpy.diagonal(gamma, offset=p*num_candidates)
                diag.setflags(write=True)
                diag.fill(0)
                        
                diag=numpy.diagonal(gamma, offset=-p*num_candidates)
                diag.setflags(write=True)
                diag.fill(0)
                        
        #Now correct the diagonal blocks: Diagonal matrix with marginals = 1/m
        for j in range(validDocs):
            currentStart=j*num_candidates
            currentEnd=(j+1)*num_candidates
            gamma[currentStart:currentEnd, currentStart:currentEnd]=0
            numpy.fill_diagonal(gamma, 1.0/num_candidates)

    gammaInv=scipy.linalg.pinv(gamma)
    return (num_candidates, gammaInv)
    
    
#NonUniformGamma(...) computes a Gamma_pinv matrix for non-uniform exploration
#num_candidates: (int) Number of candidates, m
#decay: (double) Decay factor. Doc Selection Prob \propto exp2(-decay * floor[ log2(rank) ])
#ranking_size: (int) Size of slate, l
#allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def NonUniformGamma(num_candidates, decay, ranking_size, allow_repetitions):
    validDocs=ranking_size
    if not allow_repetitions:
        validDocs=min(ranking_size, num_candidates)

    multinomial=numpy.arange(1, num_candidates+1, dtype=numpy.float64)
    multinomial=numpy.exp2((-decay)*numpy.floor(numpy.log2(multinomial)))
    
    for i in range(1,num_candidates):
        prevVal=multinomial[i-1]
        currVal=multinomial[i]
        if numpy.isclose(currVal, prevVal):
            multinomial[i]=prevVal
    
    gamma=None
    if num_candidates==1:
        gamma=numpy.ones((num_candidates*validDocs, num_candidates*validDocs), dtype=numpy.longdouble)
    else:
        if allow_repetitions:
            offDiagonal=numpy.outer(multinomial, multinomial)
            gamma=numpy.tile(offDiagonal, (validDocs, validDocs))
            for j in range(validDocs):
                currentStart=j*num_candidates
                currentEnd=(j+1)*num_candidates
                gamma[currentStart:currentEnd, currentStart:currentEnd]=numpy.diag(multinomial)
        else:
            gammaVals=GammaDP.GammaCalculator(multinomial.tolist(), validDocs)
            gamma=numpy.diag(numpy.ravel(gammaVals.unitMarginals))
            
            for p in range(validDocs):
                for q in range(p+1, validDocs):
                    pairMarginals=gammaVals.pairwiseMarginals[(p,q)]
                    currentRowStart=p*num_candidates
                    currentRowEnd=(p+1)*num_candidates
                    currentColumnStart=q*num_candidates
                    currentColumnEnd=(q+1)*num_candidates
                    gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd]=pairMarginals
                    gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd]=pairMarginals.T
    
    normalizer=numpy.sum(multinomial, dtype=numpy.longdouble)
    multinomial=multinomial/normalizer

    gammaInv=scipy.linalg.pinv(gamma)
    return (num_candidates, multinomial, gammaInv)
    

class RecursiveSlateEval:
    def __init__(self, scores):
        self.m=scores.shape[0]
        self.l=scores.shape[1]
        self.scores=scores
        self.sortedIndices=numpy.argsort(scores, axis=0)
        self.bestSoFar=None
        self.bestSlate=None
        self.counter=0
        self.upperPos=numpy.amax(scores, axis=0)
        self.eval_slate([], 0.0)
        print(self.m, self.counter, flush=True)
        
    def eval_slate(self, slate_prefix, prefix_value):
        currentPos=len(slate_prefix)
        if currentPos==self.l:
            self.counter+=1
            if self.bestSoFar is None or prefix_value > self.bestSoFar:
                self.bestSoFar=prefix_value
                self.bestSlate=slate_prefix
            return
        
        docSet=set(slate_prefix)
        bestFutureVal=0.0
        if currentPos < self.l:
            bestFutureVal=self.upperPos[currentPos:].sum()
        delta=prefix_value+bestFutureVal
        for i in range(self.m):
            currentDoc=self.sortedIndices[-1-i, currentPos]
            if currentDoc in docSet:
                continue
            currentVal=self.scores[currentDoc, currentPos]
            if self.bestSoFar is None or ((currentVal+delta) > self.bestSoFar):
                self.eval_slate(slate_prefix + [currentDoc], prefix_value+currentVal)
            else:
                break
            
class Policy:
    #dataset: (Datasets) Must be initialized using Datasets.loadTxt(...)/loadNpz(...)
    #allow_repetitions: (bool) If true, the policy predicts rankings with repeated documents
    def __init__(self, dataset, allow_repetitions):
        self.dataset=dataset
        self.allowRepetitions=allow_repetitions
        self.name=None
        ###All sub-classes of Policy should supply a predict method
        ###Requires: (int) query_id; (int) ranking_size.
        ###Returns: list[int],length=min(ranking_size,docsPerQuery[query_id]) ranking


class L2RPolicy(Policy):
    def __init__(self, dataset, ranking_size, model_type, greedy_select, cross_features):
        Policy.__init__(self, dataset, False)
        self.rankingSize=ranking_size
        self.numDocFeatures=dataset.features[0].shape[1]
        self.modelType=model_type
        self.crossFeatures=cross_features
        self.hyperParams=numpy.logspace(0,2,num=5,base=10).tolist()
        if self.modelType=='tree' or self.modelType=='gbrt':
            self.tree=None
        else:
            self.policyParams=None

        self.greedy=greedy_select
        
        self.numFeatures=self.numDocFeatures+self.rankingSize 
        if self.crossFeatures:
            self.numFeatures+=self.numDocFeatures*self.rankingSize
        print("L2RPolicy:init [INFO] Dataset:", dataset.name, flush=True)

    def createFeature(self, docFeatures, position):
        currFeature=numpy.zeros(self.numFeatures, dtype=numpy.float64)
        currFeature[0:self.numDocFeatures]=docFeatures
        currFeature[self.numDocFeatures+position]=1
        if self.crossFeatures:
            currFeature[self.numDocFeatures+self.rankingSize+position*self.numDocFeatures: \
                        self.numDocFeatures+self.rankingSize+(position+1)*self.numDocFeatures]=docFeatures

        return currFeature.reshape(1,-1)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]
        validDocs=min(allowedDocs, self.rankingSize)

        allScores=numpy.zeros((allowedDocs, validDocs), dtype=numpy.float64)
        allFeatures=self.dataset.features[query_id].toarray()
        
        for doc in range(allowedDocs):
            docID=doc
            if self.dataset.mask is not None:
                docID=self.dataset.mask[query_id][doc]
            for pos in range(validDocs):
                currFeature=self.createFeature(allFeatures[docID,:], pos)

                if self.modelType=='tree' or self.modelType=='gbrt':
                    allScores[doc, pos]=self.tree.predict(currFeature)
                else:
                    allScores[doc, pos]=currFeature.dot(self.policyParams)

        tieBreaker=1e-14*numpy.random.random((allowedDocs, validDocs))
        allScores+=tieBreaker
        upperBound=numpy.amax(allScores, axis=0)
        
        producedRanking=None
        if self.greedy:
            
            producedRanking=numpy.empty(validDocs, dtype=numpy.int32)
            currentVal=0.0
            for i in range(validDocs):
                maxIndex=numpy.argmax(allScores)
                chosenDoc,chosenPos = numpy.unravel_index(maxIndex, allScores.shape)
                currentVal+=allScores[chosenDoc, chosenPos]
                if self.dataset.mask is None:
                    producedRanking[chosenPos]=chosenDoc
                else:
                    producedRanking[chosenPos]=self.dataset.mask[query_id][chosenDoc]
                
                allScores[chosenDoc,:] = float('-inf')
                allScores[:,chosenPos] = float('-inf')
            
            self.debug=upperBound.sum()-currentVal
        else:
            slateScorer=RecursiveSlateEval(allScores)
            if self.dataset.mask is None:
                producedRanking=numpy.array(slateScorer.bestSlate)
            else:
                producedRanking=self.dataset.mask[slateScorer.bestSlate]
                
            self.debug=upperBound.sum()-slateScorer.bestSoFar
            del slateScorer
            
        del allFeatures
        del allScores
        
        return producedRanking

    def train(self, dataset, targets, hyper_params):
        numQueries=len(dataset.docsPerQuery)
        validDocs=numpy.minimum(dataset.docsPerQuery, self.rankingSize)
        queryDocPosTriplets=numpy.dot(dataset.docsPerQuery, validDocs)
        designMatrix=numpy.zeros((queryDocPosTriplets, self.numFeatures), dtype=numpy.float32, order='F')
        regressionTargets=numpy.zeros(queryDocPosTriplets, dtype=numpy.float64, order='C')
        sampleWeights=numpy.zeros(queryDocPosTriplets, dtype=numpy.float32)
        currID=-1
        for i in range(numQueries):
            numAllowedDocs=dataset.docsPerQuery[i]
            currValidDocs=validDocs[i]
            allFeatures=dataset.features[i].toarray()
            
            for doc in range(numAllowedDocs):
                docID=doc
                if dataset.mask is not None:
                    docID=dataset.mask[i][doc]
                    
                for j in range(currValidDocs):
                    currID+=1

                    designMatrix[currID,:]=self.createFeature(allFeatures[docID,:], j)
                    regressionTargets[currID]=targets[i][j,doc] 
                    sampleWeights[currID]=1.0/(numAllowedDocs * currValidDocs)
        
        for i in targets:
            del i
        del targets
        
        print("L2RPolicy:train [LOG] Finished creating features and targets ", 
                numpy.amin(regressionTargets), numpy.amax(regressionTargets), numpy.median(regressionTargets), flush=True)
        print("L2RPolicy:train [LOG] Histogram of targets ", numpy.histogram(regressionTargets), flush=True)
        
        if self.modelType == 'gbrt':
            tree=sklearn.ensemble.GradientBoostingRegressor(learning_rate=hyper_params['lr'],
                            n_estimators=hyper_params['ensemble'], subsample=hyper_params['subsample'], max_leaf_nodes=hyper_params['leaves'], 
                            max_features=1.0, presort=False)
            tree.fit(designMatrix, regressionTargets, sample_weight=sampleWeights)
            self.tree=tree
            print("L2RPolicy:train [INFO] %s" % self.modelType, flush=True)
                
        elif self.modelType == 'ridge':
            ridgeCV=sklearn.linear_model.RidgeCV(alphas=self.hyperParams, fit_intercept=False,
                                                            normalize=False, cv=3)
            ridgeCV.fit(designMatrix, regressionTargets, sample_weight=sampleWeights)
            self.policyParams=ridgeCV.coef_
            print("L2RPolicy:train [INFO] Done. ", flush=True)
            
        else:
            print("L2RPolicy:train [ERR] %s not supported." % self.modelType, flush = True)
            sys.exit(0)
            
        print("L2R:train [INFO] Created %s predictor using dataset %s." %
                (self.modelType, dataset.name), flush = True)
                
                
class DeterministicPolicy(Policy):
    #model_type: (str) Model class to use for scoring documents
    def __init__(self, dataset, model_type, regress_gains=False, weighted_ls=False, hyper_params=None):
        Policy.__init__(self, dataset, False)
        self.modelType=model_type
        self.hyperParams={'alpha': (numpy.logspace(-3,2,num=6,base=10)).tolist()}
        if hyper_params is not None:
            self.hyperParams=hyper_params
        
        self.regressGains=regress_gains
        self.weighted=weighted_ls
        
        self.treeDepths={'max_depth': list(range(3,21,3))}
        
        #Must call train(...) to set all these members
        #before using DeterministicPolicy objects elsewhere
        self.featureList=None
        if self.modelType=='tree':
            self.tree=None
        else:
            self.policyParams=None
            
        #These members are set by predictAll(...) method
        self.savedRankingsSize=None
        self.savedRankings=None
        
        print("DeterministicPolicy:init [INFO] Dataset", dataset.name, flush=True)
    
    #feature_list: list[int],length=unmaskedFeatures; List of features that should be used for training
    #name: (str) String to help identify this DeterministicPolicy object henceforth
    def train(self, feature_list, name):
        self.featureList=feature_list
        self.name=name+'-'+self.modelType
        modelFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name
        if 'alpha' not in self.hyperParams:
            #Expecting hyper-params for GBRT; Add those hyper-params to the model file name
            modelFile=modelFile+'ensemble-'+str(self.hyperParams['ensemble'])+'_lr-'+str(self.hyperParams['lr'])+'_subsample-'+str(self.hyperParams['subsample'])+'_leaves-'+str(self.hyperParams['leaves'])
            
        if self.modelType=='tree' or self.modelType=='gbrt':
            modelFile+='.z'
        else:
            modelFile+='.npz'
            
        self.savedRankingsSize=None
        self.savedRankings=None
        
        if os.path.exists(modelFile):
            if self.modelType=='tree' or self.modelType=='gbrt':
                self.tree=joblib.load(modelFile)
                print("DeterministicPolicy:train [INFO] Using precomputed policy", modelFile, flush=True)
            else:
                with numpy.load(modelFile) as npFile:
                    self.policyParams=npFile['policyParams']
                print("DeterministicPolicy:train [INFO] Using precomputed policy", modelFile, flush=True)
                print("DeterministicPolicy:train [INFO] PolicyParams", self.policyParams,flush=True)
        else:
            numQueries=len(self.dataset.features)
        
            allFeatures=None
            allTargets=None
            print("DeterministicPolicy:train [INFO] Constructing features and targets", flush=True)
                
            if self.dataset.mask is None:
                allFeatures=scipy.sparse.vstack(self.dataset.features, format='csc')
                allTargets=numpy.hstack(self.dataset.relevances)
            else:
                temporaryFeatures=[]
                temporaryTargets=[]
                for currentQuery in range(numQueries):
                    temporaryFeatures.append(self.dataset.features[currentQuery][self.dataset.mask[currentQuery], :])
                    temporaryTargets.append(self.dataset.relevances[currentQuery][self.dataset.mask[currentQuery]])
                
                allFeatures=scipy.sparse.vstack(temporaryFeatures, format='csc')
                allTargets=numpy.hstack(temporaryTargets)
        
            if self.regressGains:
                allTargets=numpy.exp2(allTargets)-1.0
            
            allSampleWeights=None
            fitParams=None
            if self.weighted:
                allSampleWeights=numpy.array(self.dataset.docsPerQuery, dtype=numpy.float64)
                allSampleWeights=numpy.reciprocal(allSampleWeights)
                allSampleWeights=numpy.repeat(allSampleWeights, self.dataset.docsPerQuery)    
                fitParams={'sample_weight': allSampleWeights}
            
            #Restrict features to only the unmasked features
            if self.featureList is not None:
                print("DeterministicPolicy:train [INFO] Masking unused features. Remaining feature size", 
                    len(feature_list), flush=True)
                allFeatures = allFeatures[:, self.featureList]
        
            print("DeterministicPolicy:train [INFO] Beginning training", self.modelType, flush=True)
            if self.modelType=='tree':
                treeCV=sklearn.model_selection.GridSearchCV(sklearn.tree.DecisionTreeRegressor(criterion="mse",
                                                        splitter="random", min_samples_split=4, 
                                                        min_samples_leaf=4, presort=False),
                                param_grid=self.treeDepths,
                                scoring=None, fit_params=fitParams, n_jobs=-2,
                                iid=True, cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
                                error_score='raise', return_train_score=False)
                            
                treeCV.fit(allFeatures, allTargets)
                self.tree=treeCV.best_estimator_
                print("DeterministicPolicy:train [INFO] Done. Best depth", 
                            treeCV.best_params_['max_depth'], flush=True)
                joblib.dump(self.tree, modelFile, compress=9, protocol=-1)
            
            elif self.modelType=='lasso':
                lassoCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(fit_intercept=False,
                                                        normalize=False, precompute=False, copy_X=False, 
                                                        max_iter=3000, tol=1e-4, warm_start=False, positive=False,
                                                        random_state=None, selection='random'),
                                param_grid=self.hyperParams,
                                scoring=None, fit_params=fitParams, n_jobs=-2,
                                iid=True, cv=5, refit=True, verbose=0, pre_dispatch="1*n_jobs",
                                error_score='raise', return_train_score=False)
                                
                lassoCV.fit(allFeatures, allTargets)
                self.policyParams=lassoCV.best_estimator_.coef_
                print("DeterministicPolicy:train [INFO] Done. CVAlpha", lassoCV.best_params_['alpha'], flush=True)
                print("DeterministicPolicy:train [INFO] PolicyParams", self.policyParams,flush=True)
                numpy.savez_compressed(modelFile, policyParams=self.policyParams)
        
            elif self.modelType == 'ridge':
                ridgeCV=sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(fit_intercept=False,
                                                                                    normalize=False, copy_X=False,
                                                                                    max_iter=3000, tol=1e-4, random_state=None),
                                                         param_grid=self.hyperParams,
                                                         n_jobs=-2, fit_params=fitParams,
                                                         iid=True, cv=3, refit=True, verbose=0, pre_dispatch='1*n_jobs')
                ridgeCV.fit(allFeatures, allTargets)
                self.policyParams=ridgeCV.best_estimator_.coef_
                print("DeterministicPolicy:train [INFO] Done. CVAlpha", ridgeCV.best_params_['alpha'], flush=True)
            elif self.modelType=='gbrt':
                tree=sklearn.ensemble.GradientBoostingRegressor(learning_rate=self.hyperParams['lr'],
                            n_estimators=self.hyperParams['ensemble'], subsample=self.hyperParams['subsample'], max_leaf_nodes=self.hyperParams['leaves'], 
                            max_features=1.0, presort=False)
                tree.fit(allFeatures, allTargets, sample_weight=allSampleWeights)
                self.tree=tree
                print("DeterministicPolicy:train [INFO] Done.", flush=True)
                joblib.dump(self.tree, modelFile, compress=9, protocol=-1)
            
            else:
                print("DeterministicPolicy:train [ERR] %s not supported." % self.modelType, flush=True)
                sys.exit(0)
    
    #query_id: (int) Query ID in self.dataset
    #ranking_size: (int) Size of ranking. Returned ranking length is min(ranking_size,docsPerQuery[query_id])
    #                       Use ranking_size=-1 to rank all available documents for query_id
    def predict(self, query_id, ranking_size):
        if self.savedRankingsSize is not None and self.savedRankingsSize==ranking_size:
            return self.savedRankings[query_id]
        
        allowedDocs=self.dataset.docsPerQuery[query_id]
        validDocs=ranking_size
        if ranking_size <= 0 or validDocs > allowedDocs:
            validDocs=allowedDocs
        
        currentFeatures=None
        if self.dataset.mask is None:
            if self.featureList is not None:
                currentFeatures=self.dataset.features[query_id][:, self.featureList]
            else:
                currentFeatures=self.dataset.features[query_id]
            
        else:
            currentFeatures=self.dataset.features[query_id][self.dataset.mask[query_id], :]
            if self.featureList is not None:
                currentFeatures=currentFeatures[:, self.featureList]
        
        allDocScores=None
        if self.modelType=='tree':
            allDocScores=self.tree.predict(currentFeatures)
        elif self.modelType=='gbrt':
            allDocScores=self.tree.predict(currentFeatures.toarray())
        else:
            allDocScores=currentFeatures.dot(self.policyParams)
            
        tieBreaker=numpy.random.random(allDocScores.size)
        sortedDocScores=numpy.lexsort((tieBreaker,-allDocScores))[0:validDocs]
        if self.dataset.mask is None:
            return sortedDocScores
        else:
            return self.dataset.mask[query_id][sortedDocScores]
    
    #ranking_size: (int) Size of ranking. Returned ranking length is min(ranking_size,docsPerQuery[query_id])
    #                       Use ranking_size=-1 to rank all available documents for query_id
    def predictAll(self, ranking_size):
        if self.savedRankingsSize is not None and self.savedRankingsSize==ranking_size:
            return
            
        numQueries=len(self.dataset.features)
        predictedRankings=[]
        for i in range(numQueries):
            predictedRankings.append(self.predict(i, ranking_size))
                
            if i%100==0:
                print(".", end="", flush=True)
                
        self.savedRankingsSize=ranking_size
        self.savedRankings=predictedRankings
        print("", flush=True)
        print("DeterministicPolicy:predictAll [INFO] Generated all predictions for %s using policy: " %
                self.dataset.name, self.name, flush=True)
        
    #num_allowed_docs: (int) Filters the dataset where the max docs per query is num_allowed_docs.
    #                        Uses policyParams to rank and filter the original document set.
    def filterDataset(self, num_allowed_docs):
        self.savedRankingsSize=None
        self.savedRankings=None
        
        numQueries=len(self.dataset.docsPerQuery)
        
        self.dataset.name=self.dataset.name+'-filt('+self.name+'-'+str(num_allowed_docs)+')'
        
        newMask = []
        for i in range(numQueries):
            producedRanking=self.predict(i, num_allowed_docs)
            self.dataset.docsPerQuery[i]=numpy.shape(producedRanking)[0]
            newMask.append(producedRanking)
            if i%100==0:
                print(".", end="", flush=True)
                
        self.dataset.mask=newMask
        print("", flush=True)
        print("DeterministicPolicy:filteredDataset [INFO] New Name", self.dataset.name, "\t MaxNumDocs", num_allowed_docs, flush=True)

        
class UniformPolicy(Policy):
    def __init__(self, dataset, allow_repetitions):
        Policy.__init__(self, dataset, allow_repetitions)
        self.name='Unif-'
        if allow_repetitions:
            self.name+='Rep'
        else:
            self.name+='NoRep'
    
        #These members are set on-demand by setupGamma(...)
        self.gammas=None
        self.gammaRankingSize=None
        
        print("UniformPolicy:init [INFO] Dataset: %s AllowRepetitions:" % dataset.name,
                        allow_repetitions, flush=True)
    
    #ranking_size: (int) Size of ranking.
    def setupGamma(self, ranking_size):
        if self.gammaRankingSize is not None and self.gammaRankingSize==ranking_size:
            print("UniformPolicy:setupGamma [INFO] Gamma has been pre-computed for this ranking_size. Size of Gamma cache:", len(self.gammas), flush=True)
            return
        
        gammaFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name+'_'+str(ranking_size)+'.z'
        if os.path.exists(gammaFile):
            self.gammas=joblib.load(gammaFile)
            self.gammaRankingSize=ranking_size
            print("UniformPolicy:setupGamma [INFO] Using precomputed gamma", gammaFile, flush=True)
            
        else:
            self.gammas={}
            self.gammaRankingSize=ranking_size
            
            candidateSet=set(self.dataset.docsPerQuery)
            
            responses=joblib.Parallel(n_jobs=-2, verbose=50)(joblib.delayed(UniformGamma)(i, ranking_size, self.allowRepetitions) for i in candidateSet)
            
            for tup in responses:
                self.gammas[tup[0]]=tup[1]
            
            joblib.dump(self.gammas, gammaFile, compress=9, protocol=-1)
            print("", flush=True)
            print("UniformPolicy:setupGamma [INFO] Finished creating Gamma_pinv cache. Size", len(self.gammas), flush=True)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]    
        
        validDocs=ranking_size
        if ranking_size < 0 or ((not self.allowRepetitions) and (validDocs > allowedDocs)):
            validDocs=allowedDocs
            
        producedRanking=None
        if self.allowRepetitions:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=True)
        else:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=False)
                                
        if self.dataset.mask is None:
            return producedRanking
        else:
            return self.dataset.mask[query_id][producedRanking]
        

class NonUniformPolicy(Policy):
    def __init__(self, deterministic_policy, dataset, allow_repetitions, decay):
        Policy.__init__(self, dataset, allow_repetitions)
        self.decay = decay
        self.policy = deterministic_policy
        self.name='NonUnif-'
        if allow_repetitions:
            self.name+='Rep'
        else:
            self.name+='NoRep'
        self.name += '(' + deterministic_policy.name + ';' + str(decay) + ')'
        
        #These members are set on-demand by setupGamma
        self.gammas=None
        self.multinomials=None
        self.gammaRankingSize=None
        
        print("NonUniformPolicy:init [INFO] Dataset: %s AllowRepetitions:" % dataset.name,
                        allow_repetitions, "\t Decay:", decay, flush=True)
    
    
    def setupGamma(self, ranking_size):
        if self.gammaRankingSize is not None and self.gammaRankingSize==ranking_size:
            print("NonUniformPolicy:setupGamma [INFO] Gamma has been pre-computed for this ranking_size. Size of Gamma cache:", len(self.gammas), flush=True)
            return
        
        gammaFile=Settings.DATA_DIR+self.dataset.name+'_'+self.name+'_'+str(ranking_size)+'.z'
        if os.path.exists(gammaFile):
            self.gammas, self.multinomials=joblib.load(gammaFile)
            self.gammaRankingSize=ranking_size
            print("NonUniformPolicy:setupGamma [INFO] Using precomputed gamma", gammaFile, flush=True)
            
        else:
            self.gammas={}
            self.multinomials={}
            self.gammaRankingSize=ranking_size
            
            candidateSet=set(self.dataset.docsPerQuery)
            responses=joblib.Parallel(n_jobs=-2, verbose=50)(joblib.delayed(NonUniformGamma)(i, self.decay, ranking_size, self.allowRepetitions) for i in candidateSet)
            
            for tup in responses:
                self.gammas[tup[0]]=tup[2]
                self.multinomials[tup[0]]=tup[1]
            
            joblib.dump((self.gammas, self.multinomials), gammaFile, compress=9, protocol=-1)
            print("", flush=True)
            print("NonUniformPolicy:setupGamma [INFO] Finished creating Gamma_pinv cache. Size", len(self.gammas), flush=True)

        self.policy.predictAll(-1)

    def predict(self, query_id, ranking_size):
        allowedDocs=self.dataset.docsPerQuery[query_id]    
        underlyingRanking=self.policy.predict(query_id, -1)
            
        validDocs=ranking_size
        if ranking_size < 0 or ((not self.allowRepetitions) and (validDocs > allowedDocs)):
            validDocs=allowedDocs
            
        currentDistribution=self.multinomials[allowedDocs]
        producedRanking=None
        if self.allowRepetitions:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=True, p=currentDistribution)
        else:
            producedRanking=numpy.random.choice(allowedDocs, size=validDocs,
                                replace=False, p=currentDistribution)
                                
        return underlyingRanking[producedRanking]
```

```python id="Ts3P5f8Hg-ov"
M=100
L=10
resetSeed=387

mslrData=Datasets.Datasets()
mslrData.loadNpz(Settings.DATA_DIR+'MSLR/mslr')

anchorURLFeatures, bodyTitleDocFeatures=Settings.get_feature_sets("MSLR")

numpy.random.seed(resetSeed)
detLogger=DeterministicPolicy(mslrData, 'tree')
detLogger.train(anchorURLFeatures, 'url')

detLogger.filterDataset(M)

filteredDataset=detLogger.dataset
del mslrData
del detLogger

uniform=UniformPolicy(filteredDataset, False)
uniform.setupGamma(L)
del uniform

numpy.random.seed(resetSeed)
loggingPolicyTree=DeterministicPolicy(filteredDataset, 'tree')
loggingPolicyTree.train(anchorURLFeatures, 'url')
        
numpy.random.seed(resetSeed)
targetPolicyTree=DeterministicPolicy(filteredDataset, 'tree')
targetPolicyTree.train(bodyTitleDocFeatures, 'body')

numpy.random.seed(resetSeed)
loggingPolicyLinear=DeterministicPolicy(filteredDataset, 'lasso')
loggingPolicyLinear.train(anchorURLFeatures, 'url')

numpy.random.seed(resetSeed)
targetPolicyLinear=DeterministicPolicy(filteredDataset, 'lasso')
targetPolicyLinear.train(bodyTitleDocFeatures, 'body')

numQueries=len(filteredDataset.docsPerQuery)

TTtau=[]
TToverlap=[]
TLtau=[]
TLoverlap=[]
LTtau=[]
LToverlap=[]
LLtau=[]
LLoverlap=[]
LogLogtau=[]
LogLogoverlap=[]
TargetTargettau=[]
TargetTargetoverlap=[]

def computeTau(ranking1, ranking2):
    rank1set=set(ranking1)
    rank2set=set(ranking2)
    documents=rank1set | rank2set
    rankingSize=len(rank1set)
    
    newRanking1=numpy.zeros(len(documents), dtype=numpy.int)
    newRanking2=numpy.zeros(len(documents), dtype=numpy.int)
    
    for docID, doc in enumerate(documents):
        if doc not in rank1set:
            newRanking1[docID]=rankingSize + 1
            newRanking2[docID]=ranking2.index(doc)
        elif doc not in rank2set:
            newRanking2[docID]=rankingSize + 1
            newRanking1[docID]=ranking1.index(doc)
        else:
            newRanking1[docID]=ranking1.index(doc)
            newRanking2[docID]=ranking2.index(doc)
        
    return scipy.stats.kendalltau(newRanking1, newRanking2)[0], 1.0*len(rank1set&rank2set)/rankingSize

numpy.random.seed(resetSeed)    
for currentQuery in range(numQueries):
    if filteredDataset.docsPerQuery[currentQuery]<4:
        continue
        
    logTreeRanking=loggingPolicyTree.predict(currentQuery, L).tolist()
    logLinearRanking=loggingPolicyLinear.predict(currentQuery, L).tolist()
    
    targetTreeRanking=targetPolicyTree.predict(currentQuery, L).tolist()
    targetLinearRanking=targetPolicyLinear.predict(currentQuery, L).tolist()
    
    tau, overlap=computeTau(logTreeRanking, targetTreeRanking)
    TTtau.append(tau)
    TToverlap.append(overlap)
    
    tau, overlap=computeTau(logTreeRanking, targetLinearRanking)
    TLtau.append(tau)
    TLoverlap.append(overlap)
    
    tau, overlap=computeTau(logLinearRanking, targetTreeRanking)
    LTtau.append(tau)
    LToverlap.append(overlap)
    
    tau, overlap=computeTau(logLinearRanking, targetLinearRanking)
    LLtau.append(tau)
    LLoverlap.append(overlap)
    
    tau, overlap=computeTau(logLinearRanking, logTreeRanking)
    LogLogtau.append(tau)
    LogLogoverlap.append(overlap)
    
    tau, overlap=computeTau(targetLinearRanking, targetTreeRanking)
    TargetTargettau.append(tau)
    TargetTargetoverlap.append(overlap)
    
    if len(TTtau) % 100 == 0:
        print(".", end="", flush=True)

TTtau=numpy.array(TTtau)
TLtau=numpy.array(TLtau)
LTtau=numpy.array(LTtau)
LLtau=numpy.array(LLtau)
LogLogtau=numpy.array(LogLogtau)
TargetTargettau=numpy.array(TargetTargettau)

TToverlap=numpy.array(TToverlap)
TLoverlap=numpy.array(TLoverlap)
LToverlap=numpy.array(LToverlap)
LLoverlap=numpy.array(LLoverlap)
LogLogoverlap=numpy.array(LogLogoverlap)
TargetTargetoverlap=numpy.array(TargetTargetoverlap)

print("", flush=True)    
print("TTtau", numpy.amax(TTtau), numpy.amin(TTtau), numpy.mean(TTtau), numpy.std(TTtau), numpy.median(TTtau), len(numpy.where(TTtau > 0.99)[0]))
print("TToverlap", numpy.amax(TToverlap), numpy.amin(TToverlap), numpy.mean(TToverlap), numpy.std(TToverlap), numpy.median(TToverlap), len(numpy.where(TToverlap > 0.99)[0]))
print("TLtau", numpy.amax(TLtau), numpy.amin(TLtau), numpy.mean(TLtau), numpy.std(TLtau), numpy.median(TLtau), len(numpy.where(TLtau > 0.99)[0]))
print("TLoverlap", numpy.amax(TLoverlap), numpy.amin(TLoverlap), numpy.mean(TLoverlap), numpy.std(TLoverlap), numpy.median(TLoverlap), len(numpy.where(TLoverlap > 0.99)[0]))
print("LTtau", numpy.amax(LTtau), numpy.amin(LTtau), numpy.mean(LTtau), numpy.std(LTtau), numpy.median(LTtau), len(numpy.where(LTtau > 0.99)[0]))
print("LToverlap", numpy.amax(LToverlap), numpy.amin(LToverlap), numpy.mean(LToverlap), numpy.std(LToverlap), numpy.median(LToverlap), len(numpy.where(LToverlap > 0.99)[0]))
print("LLtau", numpy.amax(LLtau), numpy.amin(LLtau), numpy.mean(LLtau), numpy.std(LLtau), numpy.median(LLtau), len(numpy.where(LLtau > 0.99)[0]))
print("LLoverlap", numpy.amax(LLoverlap), numpy.amin(LLoverlap), numpy.mean(LLoverlap), numpy.std(LLoverlap), numpy.median(LLoverlap), len(numpy.where(LLoverlap > 0.99)[0]))
print("LogLogtau", numpy.amax(LogLogtau), numpy.amin(LogLogtau), numpy.mean(LogLogtau), numpy.std(LogLogtau), numpy.median(LogLogtau), len(numpy.where(LogLogtau > 0.99)[0]))
print("LogLogoverlap", numpy.amax(LogLogoverlap), numpy.amin(LogLogoverlap), numpy.mean(LogLogoverlap), numpy.std(LogLogoverlap), numpy.median(LogLogoverlap), len(numpy.where(LogLogoverlap > 0.99)[0]))
print("TargetTargettau", numpy.amax(TargetTargettau), numpy.amin(TargetTargettau), numpy.mean(TargetTargettau), numpy.std(TargetTargettau), numpy.median(TargetTargettau), len(numpy.where(TargetTargettau > 0.99)[0]))
print("TargetTargetoverlap", numpy.amax(TargetTargetoverlap), numpy.amin(TargetTargetoverlap), numpy.mean(TargetTargetoverlap), numpy.std(TargetTargetoverlap), numpy.median(TargetTargetoverlap), len(numpy.where(TargetTargetoverlap > 0.99)[0]))
```

<!-- #region id="OarlWfFjhQBo" -->
## Optimization
<!-- #endregion -->

<!-- #region id="c-VnRUUzhQkx" -->
Script for semi-synthetic optimization runs
<!-- #endregion -->

```python id="QROXnqZJhZ6v"
# import Datasets
# import argparse
# import Settings
# import sys
# import os
# import numpy
# import Policy
# import Metrics
```

```python id="dd-RLGN8hcmM"

```
