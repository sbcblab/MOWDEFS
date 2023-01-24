import os
import pathlib
import numpy as np
from sklearn.metrics import confusion_matrix
import math
import RR_utils

from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


### Classifiers worth using
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier       ### DecisionTreeClassifier() --> DT
from sklearn.neural_network import MLPClassifier      ### MLPClassifier()          --> MLP
from sklearn.neighbors import KNeighborsClassifier    ### KNeighborsClassifier()   --> KNN
from sklearn.linear_model import LogisticRegression   ### LogisticRegression()     --> LR
from sklearn.ensemble import RandomForestClassifier   ### RandomForestClassifier() --> RF

from pycm import *


def error_rate2(xtrain, ytrain, x, opts, out_fold, i, df, cfg, run, statistics):
    np.seterr(invalid='ignore')
    k     = opts['k']
    # fold  = opts['fold']
    # xt    = fold['xt']
    # yt    = fold['yt']
    # xv    = fold['xv']
    # yv    = fold['yv']
    C     = opts['Classes']
    max_iter = opts['T']

    feat  = opts['feat']
    label = opts['label']
    X     = feat[:, x==1]
    # print(x)
    # print(len(x))
    # print(len(feat[0]))
    # print(len(X[0]))
    # print('\n\n')

    tipos = np.sort(df[cfg.class_label].unique())
    #print(X)
    Y     = label    
    #print(X)
    #print(Y)
    kf = StratifiedKFold(n_splits = 5, random_state=145, shuffle=True)
    #model = KNeighborsClassifier(n_neighbors = k)
    #model = LogisticRegression()
    #model = MLPClassifier()
    #model = SVC()  
    #model = SVC(kernel='rbf',gamma='scale',C=5000, probability=True)
    #model = DecisionTreeClassifier()
    model = RandomForestClassifier()
    count = 0
    cnf_matrix_aux = np.zeros((int(C),int(C)))
    TN = 0
    FP = 0
    FN = 0
    TP = 0
    Accuracy = 0
    Precision = 0
    Recall = 0
    F1Score = 0
    Specificity = 0
    LRP = 0
    LRM = 0      
    for train_indices, test_indices in kf.split(X, Y):
        train_X, train_y = X[train_indices], Y[train_indices]
        test_X, test_y = X[test_indices], Y[test_indices]
        pred=model.fit(train_X, train_y).predict(test_X)
        #print(train_y, test_y)
        tn,fp,fn,tp = (confusion_matrix(test_y,pred,labels=tipos).ravel())
        cnf_matrix_pycm = ConfusionMatrix(test_y, pred)
        if count == 0:
            cnf_matrix_aux_pycm = cnf_matrix_pycm
        #print("\n", tn, fp, fn, tp)
        ##print(cnf_matrix)
        # TN[i-1] = TN[i-1] + cnf_matrix[:, 0, 0]
        # FP[i-1] = FP[i-1] + cnf_matrix[:, 1, 1]
        # FN[i-1] = FN[i-1] + cnf_matrix[:, 1, 0]
        # TP[i-1] = TP[i-1] + cnf_matrix[:, 0, 1]

            # print(TN[i-1], FP[i-1], FN[i-1], TP[i-1])

        TN = TN + tn # = TN + tn
        FP = FP + fp
        FN = FN + fn
        TP = TP + tp
        if count >= 1:
            cnf_matrix_aux_pycm = cnf_matrix_aux_pycm.combine(cnf_matrix_pycm)
        count += 1
    #print(TN,FP,FN,TP)
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    #print ("Accuracy: ", Accuracy)
    Precision = (TP/(TP+FP))
    if math.isnan(Precision) == True or math.isinf(Precision) == True:
        Precision = 0.0
    #print ("Precision: ", Precision)
    Recall = (TP/(TP+FN))
    if math.isnan(Recall) == True or math.isinf(Recall) == True:
        Recall = 0.0
    #print ("Recall: ", Recall)
    F1Score = (2*(Recall * Precision) / (Recall + Precision))
    if math.isnan(F1Score) == True or math.isinf(F1Score) == True:
        F1Score = 0.0
    #print ("F1-Score: ", F1Score)
    Specificity = (TN/(TN+FP))
    
    #print ("Specificity: ", Specificity)
    LRP = (Recall / (1- Specificity))
    if math.isinf(LRP) == True or math.isnan(LRP) == True:
        LRP = 0.0
    LRM = ((1- Recall) / Specificity)
    if math.isinf(LRM) == True or math.isnan(LRM) == True:
        LRM = 0.0
    if(i == max_iter-1):
        #arq.write("RUN,TN,FP,FN,TP,Accuracy,Precision,Recall,F1Score,Specificity,Lr+,Lr-\n")
        statistics["TN"].append(TN)
        statistics["FP"].append(FP)
        statistics["FN"].append(FN)
        statistics["TP"].append(TP)
        statistics["Accuracy"].append(Accuracy)
        statistics["Precision"].append(Precision)
        statistics["Recall"].append(Recall)
        statistics["F1Score"].append(F1Score)
        statistics["Specificity"].append(Specificity)
        statistics["LRP"].append(LRP)
        statistics["LRM"].append(LRM)
        statistics["pop"].append(x)
        #arq.write(str(i)+","+str(np.mean(TN))+","+str(np.mean(FP))+","+str(np.mean(FN))+","+str(np.mean(TP))+","+str(Accuracy)+","+str(Precision)+","+str(Recall)+","+str(F1Score)+","+str(Specificity)+","+str(LRP)+","+str(LRM)+"\n")
        #arq.close()
    #print(Accuracy, F1Score)

    #cnf_matrix_aux += cnf_matrix
    #ypred   = mdl.predict(xvalid)
    #acc     = np.sum(yvalid == ypred) / num_valid
    metric   = Accuracy

    
    return metric, cnf_matrix_aux_pycm


# Error rate & Feature size 
# Fitness Function
def fitness(xtrain, ytrain, x, opts, out_fold, i, df, cfg, run, multiobjective_fitness, statistics):
    # Parameters

    alpha    = 0.1          ### Weight of the classifier
    beta     = 1 - alpha    ### Weight of the number of features selected 
    metric = 0
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    #print(num_feat, max_feat)
    fitness_feature = num_feat/max_feat

    if num_feat == 0:
        cost  = 1
        fitness_classifier = 0
        cnf_matrix_pycm = 0 
    else:
        # Get error rate
        metric, cnf_matrix_pycm = error_rate2(xtrain, ytrain, x, opts, out_fold, i, df, cfg, run, statistics)
        #print(error, num_feat, max_feat)
        # Objective function        
        #cost  = alpha * error + beta * (num_feat / max_feat)
        fitness_classifier = metric
    
    #print(fitness_classifier, fitness_feature)
    multiobjective_fitness.append([fitness_classifier, fitness_feature])

    #print(multiobjective_fitness)

    return fitness_classifier, fitness_feature, cnf_matrix_pycm, multiobjective_fitness



















# for train_indices, test_indices in kf.split(X, Y):   ### stratified k-fold
#     #for train_indices, test_indices in loo.split(X):    ### leave one out 
        
#         train_X, train_y = X[train_indices], Y[train_indices]
#         #print(train_X, train_y)
#         test_X, test_y = X[test_indices], Y[test_indices]
#         #print(test_X, test_y)
#         pred=model.fit(train_X, train_y).predict(test_X)
#         #tn, fp, fn, tp  = confusion_matrix(test_y, pred, labels=np.sort(df[cfg.class_label].unique())).ravel()
#         cnf_matrix  = confusion_matrix(test_y, pred, labels=tipos)
#         cnf_matrix_pycm = ConfusionMatrix(test_y, pred)
#         if count == 0:
#             cnf_matrix_aux_pycm = cnf_matrix_pycm
#         #print(cnf_matrix_pycm)
#         #print(cnf_matrix)
#         # TN = TN + tn
#         # FP = FP + fp
#         # FN = FN + fn
#         # TP = TP + tp
#         cnf_matrix_aux += cnf_matrix
#         #if train_indices 
#         if count >= 1:
#             cnf_matrix_aux_pycm = cnf_matrix_aux_pycm.combine(cnf_matrix_pycm)
#         #print(cnf_matrix_aux)
#         count += 1

#     #print(cnf_matrix_aux_pycm)
#     #print(TN, FP, FN, TP)
#     #time.sleep(10)
#     #print(cnf_matrix_aux_pycm.print_matrix())
#     #print(cnf_matrix_aux)
#     #print(cnf_matrix_aux)
#     FP = cnf_matrix_aux.sum(axis=0) - np.diag(cnf_matrix_aux)  
#     FN = cnf_matrix_aux.sum(axis=1) - np.diag(cnf_matrix_aux)
#     TP = np.diag(cnf_matrix_aux)
#     TN = cnf_matrix_aux.sum() - (FP + FN + TP)
#     FP = FP.astype(float)
#     FN = FN.astype(float)
#     TP = TP.astype(float)
#     TN = TN.astype(float)
#     #print(TN, FP, FN, TP)

#     Accuracy = np.mean((TP+TN)/(TP+FP+FN+TN))
#     #print(Accuracy)
#     #print(TP, TP+FP)
#     Precision = (TP)/((TP+FP))
#     Precision[np.isnan(Precision)] = 0
#     Precision = np.mean(Precision)
#     #print(Precision)    
#     if math.isnan(Precision) == True or math.isinf(Precision) == True:
#         Precision = 0.0
#     Recall = (TP)/((TP+FN))
#     Recall[np.isnan(Recall)] = 0
#     Recall = np.mean(Recall)
#     #print(Recall)
#     if math.isnan(Recall) == True or math.isinf(Recall) == True:
#         Recall = 0.0
#     F1Score = ((2*(Recall * Precision) / (Recall + Precision)))
#     #print(F1Score)
#     if math.isnan(F1Score) == True or math.isinf(F1Score) == True:
#         F1Score = 0.0
#     Specificity = (TN)/((TN+FP))
#     Specificity[np.isnan(Specificity)] = 0
#     Specificity = np.mean(Specificity)
#     LRP = ((Recall / (1- Specificity)))
#     #print(LRP)
#     if math.isinf(LRP) == True or math.isnan(LRP) == True:
#         LRP = 0.0
#     LRM = (((1- Recall) / Specificity))
#     #print(LRM)
#     if math.isinf(LRM) == True or math.isnan(LRM) == True:
#         LRM = 0.0
















# # error rate
# def error_rate(xtrain, ytrain, x, opts):
#     np.seterr(invalid='ignore')
#     # parameters
#     k     = opts['k']
#     fold  = opts['fold']
#     xt    = fold['xt']
#     yt    = fold['yt']
#     xv    = fold['xv']
#     yv    = fold['yv']
#     C     = opts['Classes']

#     feat  = opts['feat']
#     label = opts['label']
#     X     = feat[:, x==1]
#     Y     = label
#     kf = StratifiedKFold(n_splits = 3)

#     # Number of instances
#     num_train = np.size(xt, 0)
#     num_valid = np.size(xv, 0)
#     # Define selected features
#     xtrain  = xt[:, x == 1]
#     ytrain  = yt.reshape(num_train)  
#     xvalid  = xv[:, x == 1]
#     yvalid  = yv.reshape(num_valid)  #
#     #cnf_matrix_aux = np.zeros((int(C),int(C)))
#     #print(cnf_matrix_aux)
#     # Training
#     #model     = KNeighborsClassifier(n_neighbors = k)
#     #model = DecisionTreeClassifier()
#     #model = LogisticRegression(solver='liblinear')
#     model = SVC()
#     #model.fit(xtrain, ytrain)
#     # Prediction
#     pred = model.fit(xtrain, ytrain).predict(xvalid)

#     cnf_matrix = confusion_matrix(yvalid, pred)
#     #print(cnf_matrix)
#     FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
#     FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
#     TP = np.diag(cnf_matrix)
#     TN = cnf_matrix.sum() - (FP + FN + TP)
#     #print(FP, FN, TP, TN)
#     Accuracy = np.mean((TP+TN))/np.mean((TP+FP+FN+TN))
#     Precision = np.mean(TP)/np.mean((TP+FP))
#     if math.isnan(Precision) == True or math.isinf(Precision) == True:
#         Precision = 0.0
#     Recall = np.mean(TP)/np.mean((TP+FN))
#     if math.isnan(Recall) == True or math.isinf(Recall) == True:
#         Recall = 0.0
#     F1Score = np.mean((2*(Recall * Precision) / (Recall + Precision)))
#     if math.isnan(F1Score) == True or math.isinf(F1Score) == True:
#         F1Score = 0.0
#     Specificity = np.mean(TN)/np.mean(TN+FP)
#     LRP = np.mean((Recall / 1- Specificity))
#     if math.isinf(LRP) == True or math.isnan(LRP) == True:
#         LRP = 0.0
#     LRM = np.mean((1- Recall / Specificity))
#     if math.isinf(LRM) == True or math.isnan(LRM) == True:
#         LRM = 0.0

#     print(F1Score)

#     #cnf_matrix_aux += cnf_matrix
#     #ypred   = mdl.predict(xvalid)
#     #acc     = np.sum(yvalid == ypred) / num_valid
#     error   = 1 - F1Score

    
#     return error