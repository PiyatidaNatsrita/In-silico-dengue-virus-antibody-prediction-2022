from rdkit import Chem
import re, os, sys
from collections import Counter
import math
import numpy as np
from sklearn.metrics import roc_auc_score


from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
if not isJVMStarted():
    cdk_path = '../input/d/datasets/piyatidanatsrita/nuclear-smile/cdk-2.7.1.jar'
    startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % cdk_path)
    cdk =  JPackage('org').openscience.cdk

def featsmi(fp_type, smis, size=1024, depth=6):
    fg = {
            "AP2D" : cdk.fingerprint.AtomPairs2DFingerprinter(),
            "CKD":cdk.fingerprint.Fingerprinter(size, depth),
            "CKDExt":cdk.fingerprint.ExtendedFingerprinter(size, depth),
            "CKDGraph":cdk.fingerprint.GraphOnlyFingerprinter(size, depth),
            "MACCS":cdk.fingerprint.MACCSFingerprinter(),
            "PubChem":cdk.fingerprint.PubchemFingerprinter(cdk.silent.SilentChemObjectBuilder.getInstance()),
            "Estate":cdk.fingerprint.EStateFingerprinter(),
            "KR":cdk.fingerprint.KlekotaRothFingerprinter(),
            "FP4" : cdk.fingerprint.SubstructureFingerprinter(),
            "FP4C" : cdk.fingerprint.SubstructureFingerprinter(),
            "Circle" : cdk.fingerprint.CircularFingerprinter(),
            "Hybrid" : cdk.fingerprint.HybridizationFingerprinter(),
         }
    sp = cdk.smiles.SmilesParser(cdk.DefaultChemObjectBuilder.getInstance())
    for i,smi in enumerate(smis):
        mol = sp.parseSmiles(smi)
        if fp_type == "FP4C":
            fingerprinter = fg[fp_type]
            nbit = fingerprinter.getSize()
            fp = fingerprinter.getCountFingerprint(mol)
            feat = np.array([int(fp.getCount(i)) for i in range(nbit)])           
        else:
            fingerprinter = fg[fp_type]
            nbit = fingerprinter.getSize()
            fp = fingerprinter.getFingerprint(mol)
            feat = np.array([int(fp.get(i)) for i in range(nbit)])
        if i == 0:
            featx = feat.reshape(1,-1)
        else:
            featx = np.vstack((featx, feat.reshape(1,-1)))
    return featx

def featex(smis):
    fname = []
    fused = []
    feat0 = featsmi("AP2D",smis);fname.append("AP2D");fused.append(0)
    feat1 = featsmi("CKD",smis);fname.append("CKD");fused.append(1)
    feat2 = featsmi("CKDExt",smis);fname.append("CKDExt");fused.append(2)
    feat3 = featsmi("CKDGraph",smis);fname.append("CKDGraph");fused.append(3)
    feat4 = featsmi("MACCS",smis);fname.append("MACCS");fused.append(4)
    feat5 = featsmi("PubChem",smis);fname.append("PubChem");fused.append(5)
    feat6 = featsmi("Estate",smis);fname.append("Estate");fused.append(6)
    feat7 = featsmi("KR",smis);fname.append("KR");fused.append(7)
    feat8 = featsmi("FP4",smis);fname.append("FP4");fused.append(8)
    feat9 = featsmi("FP4C",smis);fname.append("FP4C");fused.append(9)
    feat10 = featsmi("Circle",smis);fname.append("Circle");fused.append(10)
    feat11 = featsmi("Hybrid",smis);fname.append("Hybrid");fused.append(11)
    allfeat_pos = np.hstack((
                             feat0, 
                             feat1, 
                             feat2, 
                             feat3, 
                             feat4,
                             feat5, 
                             feat6, 
                             feat7, 
                             feat8, 
                             feat9,
                             feat10,
                             feat11,
                            ))
    f = []
    before = 0
    for i in fused:
        after = before + eval('feat%d.shape[1]'% (i))
        f.append(list(range(before, after)))
        before = after
        
    return allfeat_pos, f, fname 

df= pd.read_csv('DENV_AbDataset.csv') # dataset file here

com_smiles = []
i = 0
for item in df['FASTA_Com']:
    i=i+1
    if item is not None:
        Chem.MolToSmiles(Chem.MolFromFASTA(item))
        com_smiles.append(Chem.MolToSmiles(Chem.MolFromFASTA(item)))

#feature extraction
com_smiles, f, fname = featex(com_smiles)

X = com_smiles
y = df['Class'].values

#Perform a train-test split. We'll use 20% of the data to evaluate the model while training on 80%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

Xall, Xtall = X_train, X_test
y, yt = y_train,  y_test

#data normalization/feature scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(Xall)
Xall = scaler.transform(Xall)
Xtall = scaler.transform(Xtall)

X_train, X_test = Xall, Xtall
X, Xt = Xall, Xtall
y, yt = y_train,  y_test


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
    X = Xall[:,f[iii]]
    Xt = Xtall[:,f[iii]]
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

    #NB
    #clf = GaussianNB()
    #acc, sens, spec, mcc, roc = cv(clf, X,y,10) 
    #allclf.append(clf)
    #file.write(str(fname[iii])+",NB,"+str(acc)+","+str(sens)+","+str(spec)+","+str(mcc)+","+str(roc)+","+str("N/A")+"\n") 

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
  
