from rdkit import Chem
import numpy as np
import multiprocessing
import logging
import pandas as pd

#featurization & ML step
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.svm import SVC

#from PLS import PLS
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))
 
 
def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))
 
def atom_features(atom,explicit_H=False,use_chirality=False):
  from rdkit import Chem
  results = one_of_k_encoding_unk(
    atom.GetSymbol(),
    [
      'C',
      'N',
      'O',
      'S',
      'H',  # H?
      'Unknown'
    ]) + one_of_k_encoding(atom.GetDegree(),
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_of_k_encoding_unk(atom.GetHybridization(), [
              Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                  SP3D, Chem.rdchem.HybridizationType.SP3D2
            ]) + [atom.GetIsAromatic()]
  # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
  if not explicit_H:
    results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                              [0, 1, 2, 3, 4])
  if use_chirality:
    try:
      results = results + one_of_k_encoding_unk(
          atom.GetProp('_CIPCode'),
          ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
    except:
      results = results + [False, False
                          ] + [atom.HasProp('_ChiralityPossible')]

  return np.array(results) 
 
def mol2vec(mol): # creates features
    atoms = mol.GetAtoms()
    node_f= [atom_features(atom) for atom in atoms]
    return node_f

df= pd.read_csv('DENV_AbDataset.csv') # dataset file here

mol_ab = []
for i in range (len(df['FASTA_Com'])):
    mol = Chem.MolFromFASTA(df['FASTA_Com'].loc[i])
    mol_ab.append(mol)


a = np.zeros((len(mol_ab),))

ab_feature = []
for i in range (len(mol_ab)):
    ab = mol2vec(mol_ab[i])
    for i in range(len(ab)):
    	if ((ab[i][29]==1)):
    		print(i)
    ab_feature.append(ab)


ab_adj = []
for i in range(len(mol_ab)):
    p1 = [Chem.rdmolops.GetAdjacencyMatrix(mol_ab[i])+np.eye(Chem.rdmolops.GetAdjacencyMatrix(mol_ab[i]).shape[0])]
    ab_adj.append(p1)


arr_ab_feature = []
for i in range (len(ab_feature)):
    arr1 = np.asarray(ab_feature[i])
    arr_ab_feature.append(arr1)

arr_ab_adj = []
for i in range (len(ab_adj)):
    arr_adj1 = np.asarray(ab_adj[i])
    arr_ab_adj.append(arr_adj1)

for i in range (len(arr_ab_adj)):
    arr_ab_adj[i] = arr_ab_adj[i].reshape(arr_ab_feature[i].shape[0],arr_ab_feature[i].shape[0])

matmul_ab = []
for i in range (len(arr_ab_adj)):
    feature_ab = np.matmul(arr_ab_adj[i],arr_ab_feature[i]) # adjacency x feat
    matmul_ab.append(feature_ab)

mean_ab = []
for i in range (len(matmul_ab)):
    mean1 = np.mean(matmul_ab[i],axis=0).reshape(37) # pooling
    mean_ab.append(mean1)

mean_ab_arr = np.asarray(mean_ab)

atom_combine = mean_ab_arr

X = atom_combine
y = df['Class'].values

from sklearn.model_selection import train_test_split

#Perform a train-test split. We'll use 20% of the data to evaluate the model while training on 80%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

Xall, Xtall  = X_train, X_test
y, yt  = y_train, y_test
X, Xt  = X_train, X_test

import re, os, sys
from collections import Counter
import math
import numpy as np
import re
from sklearn.metrics import roc_auc_score

def cv(clf, X, y, nr_fold):
    ix = []
    for i in range(0, len(y)):
        ix.append(i)
    ix = np.array(ix)
    
    allACC = []
    allSENS = []
    allSPEC = []
    allMCC = []
    allAUC = []
    for j in range(0, nr_fold):
        train_ix = ((ix % nr_fold) != j)
        test_ix = ((ix % nr_fold) == j)
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        clf.fit(train_X, train_y)        
        p = clf.predict(test_X)
        pr = clf.predict_proba(test_X)[:,1]   
        TP=0   
        FP=0
        TN=0
        FN=0
        for i in range(0,len(test_y)):
            if test_y[i]==0 and p[i]==0:
                TP+= 1
            elif test_y[i]==0 and p[i]==1:
                FN+= 1
            elif test_y[i]==1 and p[i]==0:
                FP+= 1
            elif test_y[i]==1 and p[i]==1:
                TN+= 1
        ACC = (TP+TN)/(TP+FP+TN+FN)
        SENS = TP/(TP+FN)
        SPEC = TN/(TN+FP)
        det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        if (det == 0):            
            MCC = 0                
        else:
            MCC = ((TP*TN)-(FP*FN))/det
        AUC = roc_auc_score(test_y,pr)
        allACC.append(ACC)
        allSENS.append(SENS)
        allSPEC.append(SPEC)
        allMCC.append(MCC)
        allAUC.append(AUC)
    return np.mean(allACC),np.mean(allSENS),np.mean(allSPEC),np.mean(allMCC),np.mean(allAUC)

def test(clf, X, y, Xt, yt):
    train_X, test_X = X, Xt
    train_y, test_y = y, yt
    clf.fit(train_X, train_y)        
    p = clf.predict(test_X)
    pr = clf.predict_proba(test_X)[:,1]   
    TP=0   
    FP=0
    TN=0
    FN=0
    for i in range(0,len(test_y)):
        if test_y[i]==0 and p[i]==0:
            TP+= 1
        elif test_y[i]==0 and p[i]==1:
            FN+= 1
        elif test_y[i]==1 and p[i]==0:
            FP+= 1
        elif test_y[i]==1 and p[i]==1:
            TN+= 1
    ACC = (TP+TN)/(TP+FP+TN+FN)
    SENS = TP/(TP+FN)
    SPEC = TN/(TN+FP)
    det = math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    if (det == 0):            
        MCC = 0                
    else:
        MCC = ((TP*TN)-(FP*FN))/det
    AUC = roc_auc_score(test_y,pr)
    def fcount(string, substr):
       count = 0
       pos = 0
       while(True):
           pos = string.find(substr , pos)
           if pos > -1:
               count = count + 1
               pos += 1
           else:
               break
       return count
    return ACC, SENS, SPEC, MCC, AUC

file = open('PLS.py','w')
file.write('import numpy as np'+"\n")
file.write('from sklearn.cross_decomposition import PLSRegression'+"\n")
file.write('from sklearn.base import BaseEstimator, ClassifierMixin'+"\n")
file.write('class PLS(BaseEstimator, ClassifierMixin):'+"\n")
file.write('    def __init__(self):'+"\n")
file.write('        self.clf = PLSRegression(n_components=2)'+"\n")
file.write('    def fit(self, X, y):'+"\n")
file.write('        self.clf.fit(X,y)'+"\n")
file.write('        return self'+"\n")
file.write('    def predict(self, X):'+"\n")
file.write('        pr = [np.round(np.abs(item[0])) for item in self.clf.predict(X)]'+"\n")
file.write('        return pr'+"\n")
file.write('    def predict_proba(self, X):'+"\n")
file.write('        p_all = []'+"\n")
file.write('        p_all.append([1-np.abs(item[0]) for item in self.clf.predict(X)])'+"\n")
file.write('        p_all.append([np.abs(item[0]) for item in self.clf.predict(X)])'+"\n")
file.write('        return np.transpose(np.array(p_all))'+"\n")
file.close()

from PLS import PLS
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

for iii in range(len(fname)):
    X = Xall
    Xt = Xtall
    allclf = []
    file = open("11classifier_cv.csv", "a")

    #SVM
    param = [1,2,4,8,16,32]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = SVC(C=param[i], random_state=0, probability=True)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)
    allclf.append(SVC(C=param[choose], random_state=0, probability=True).fit(X,y))
    file.write(str(fname[iii])+",SVM,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #LinearSVC
    param = [1,2,4,8,16,32]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf =  SVC(C=param[i], kernel='linear',random_state=0, probability=True)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)
    allclf.append(SVC(C=param[i], kernel='linear',random_state=0, probability=True).fit(X,y))
    file.write(str(fname[iii])+",LN,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #RF
    param = [20, 50, 100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = RandomForestClassifier(n_estimators=param[i], random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)
    allclf.append(RandomForestClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
    file.write(str(fname[iii])+",RF,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #E-Tree
    param = [20, 50, 100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = ExtraTreesClassifier(n_estimators=param[i], random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)
    allclf.append(ExtraTreesClassifier(n_estimators=param[choose], random_state=0).fit(X,y))
    file.write(str(fname[iii])+",ET,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #XGBoost
    param = [20, 50, 100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = XGBClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)  
    allclf.append(XGBClassifier(n_estimators=param[i],learning_rate=0.1, random_state=0).fit(X,y))
    file.write(str(fname[iii])+",XGB,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")  

    #MLP
    param = [20, 50, 100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):  
        clf = MLPClassifier(hidden_layer_sizes=(param[i],),random_state=0, max_iter = 10000)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)
    allclf.append(MLPClassifier(hidden_layer_sizes=(param[choose],),random_state=0, max_iter=10000).fit(X,y))
    file.write(str(fname[iii])+",MLP,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n") 


    #1NN
    clf = KNeighborsClassifier(n_neighbors=1)
    acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
    allclf.append(clf)
    file.write(str(fname[iii])+",1NN,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n")

    #DT
    clf = DecisionTreeClassifier(random_state=0)
    acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
    allclf.append(clf)
    file.write(str(fname[iii])+",DT,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

    #Logistic
    param = [0.001,0.01,0.1,1,10,100]
    acc = np.zeros(len(param)) 
    sens = np.zeros(len(param)) 
    spec = np.zeros(len(param)) 
    mcc = np.zeros(len(param)) 
    roc = np.zeros(len(param)) 
    for i in range(0,len(param)):
        clf = LogisticRegression(C=param[i], random_state=0, max_iter=10000)
        acc[i], sens[i], spec[i], mcc[i], roc[i] = cv(clf, X,y,10)
    choose = np.argmax(acc)
    allclf.append(LogisticRegression(C=param[choose], random_state=0, max_iter=10000).fit(X,y))
    file.write(str(fname[iii])+",LR,"+str(acc[choose])+","+str(sens[choose])+","+str(spec[choose])+","+str(mcc[choose])+","+str(roc[choose])+","+str(param[choose])+"\n")   

    #PLS
    clf = PLS()
    acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
    allclf.append(clf)
    file.write(str(fname[iii])+",PLS,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 


    file.close()

    ########## Test ############################
    file = open("11classifier_test.csv", "a")
    for i in range(0,len(allclf)):
        acc, sens, spec, mcc, roc = test(allclf[i], X, y, Xt, yt) 
        file.write(str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+"\n") 
    file.close()
