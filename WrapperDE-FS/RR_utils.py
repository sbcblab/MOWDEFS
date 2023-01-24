# Gabriel Dominico
# June 2022

import os
import itertools
from collections import namedtuple
import tempfile
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, KFold
import scipy.stats as stats
import math
import de
from numpy.random import rand
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

#sklearn classifiers that we can use
from sklearn.svm import SVC                           ### SVC()                    --> SVM (Support Vector Machine)
from sklearn.tree import DecisionTreeClassifier as DT      ### DecisionTreeClassifier() --> DT
from sklearn.neural_network import MLPClassifier as MLP     ### MLPClassifier()          --> MLP
from sklearn.neighbors import KNeighborsClassifier as KNN   ### KNeighborsClassifier()   --> KNN
from sklearn.linear_model import LogisticRegression as LR  ### LogisticRegression()     --> LR
from sklearn.ensemble import RandomForestClassifier  as RF  ### RandomForestClassifier() --> RF



def create_output_dir(data_file, output_folder='', dataset_format='.csv'):
    if len(output_folder) > 1:
        if output_folder[-1] != '/':
            output_folder = output_folder + '/'
    out_fold  = output_folder + os.path.basename(data_file).replace(dataset_format, '/')
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    if not os.path.exists(out_fold+"/runs"):
        os.makedirs(out_fold+"/runs")
    out_file = out_fold + os.path.basename(data_file).replace(dataset_format, '')
    return out_fold, out_file

def create_output_dir_run(data_file, output_folder='', dataset_format='.csv'):
    if len(output_folder) > 1:
        if output_folder[-1] != '/':
            output_folder = output_folder + '/'
    out_fold  = output_folder + os.path.basename(data_file).replace(dataset_format, '/')
    if not os.path.exists(out_fold):
        os.makedirs(out_fold)
    out_file = out_fold + os.path.basename(data_file).replace(dataset_format, '')
    return out_fold, out_file

def check_dataframe(df, class_label, task):
    df.columns = df.columns.str.strip()
    idx = df.index[df.index.duplicated()].unique()
    if len(idx) > 0:
        raise Exception('\nERROR: Repeated row indices in dataset: {}'.format(idx))
    idx = df.columns[df.columns.duplicated()].unique()
    if len(idx) > 0:
        raise Exception('\nERROR: Repeated column indices in dataset: {}'.format(idx))
    if class_label not in df.columns:
        raise Exception('\nERROR: Could not find class label "{}" in dataset columns: {}'.format(class_label, df.columns))
    if task == 'classification':
        df[class_label] = df[class_label].astype(str).str.strip()
    elif task == 'regression':
        df[class_label] = df[class_label].astype(float)
    return df

def get_XY(df, task, class_label):
    if df.empty:
        return None, None
    if type(class_label) is not list:
        class_label = [class_label]
    cl = class_label[0]
    dd = df
    for c in class_label:
        dd = dd.drop([c], axis=1)
    X = dd.values
    if task == 'classification':
        #print(df[cl].values)
        #Y = (df[cl]).values
        Y = pd.get_dummies(df[cl]).values
    elif task == 'regression':
        Y = df[cl].values
    return X, Y

def get_XY2(df, task, class_label):
    if df.empty:
        return None, None
    if type(class_label) is not list:
        class_label = [class_label]
    cl = class_label[0]
    dd = df
    for c in class_label:
        dd = dd.drop([c], axis=1)
    X = dd.values
    if task == 'classification':
        #print(df[cl].values)
        Y = (df[cl]).values
    elif task == 'regression':
        Y = df[cl].values
    return X, Y

def split_cv(df, task, class_label, k):
    X = df.drop([class_label],axis=1)
    y = df[class_label]
    if k > 1:
        if task == 'classification':
            skf = StratifiedKFold(n_splits=k, shuffle=True)
        elif task == 'regression':
            skf = KFold(n_splits=k, shuffle=True)
        skf.get_n_splits(X, y)
        splits = []
        for tri, tei in skf.split(X, y):
            Split = namedtuple('Split', ['tr', 'te'])
            splits.append(Split(tr=tri, te=tei))
    else:
        Split = namedtuple('Split', ['tr', 'te'])
        splits = [Split(tr=np.arange(0, len(X.index)), te=None)]
    return splits 

def split_data(df, splits, class_label, standardized, rescaled):
    tr_df = df.iloc[splits.tr]
    if splits.te is not None:
        te_df = df.iloc[splits.te]
    else:
        te_df = pd.DataFrame()

    mean_vals = None
    std_vals  = None
    if standardized:
        tr_df, mean_vals, std_vals = standardize(tr_df, class_label=class_label)
        te_df, mean_vals, std_vals = standardize(te_df, class_label=class_label, meanVals=mean_vals, stdVals=std_vals)
    min_vals = None
    max_vals = None
    if rescaled:
        tr_df, min_vals, max_vals = min_max_scaling(tr_df, class_label=class_label)
        te_df, min_vals, max_vals = min_max_scaling(te_df, class_label=class_label, minVals=min_vals, maxVals=max_vals)

    return tr_df, te_df, mean_vals, std_vals, min_vals, max_vals   
 
def get_diversity(dim, X, N):
        # moment of inertia about the centroids

        ci: list = []

        # centroids
        for d in range(dim):
            acc: float = 0.0

            for member in X:
                acc += member[d] / N

            ci.append(acc)

        Isd: float = 0.0

        for d in range(dim):
            acc: float = 0.0

            for member in X:
                acc += (member[d] - ci[d]) ** 2.0

            # average deviation
            Isd += math.sqrt(acc / (N - 1.0))

        return Isd / dim

def print_binary_population(X, N):
    for i in range(N):
        print(X[i])

#def save_final_population(X, out_fold, run):


def rand_1(i, N, V, U, X, lb, ub, dim, mutation_rate_T, crossover_rate_T):
    # Choose r1, r2, r3 randomly, but not equal to i 
    RN = np.random.permutation(N)
    for j in range(N):
        if RN[j] == i:
            RN = np.delete(RN, j)
            break

    r1 = RN[0]
    r2 = RN[1]
    r3 = RN[2]
    # mutation 
    for d in range(dim):
        #V[i,d] = X[r1,d] + F * (X[r2,d] - X[r3,d])
        V[i,d] = X[r1,d] + mutation_rate_T * (X[r2,d] - X[r3,d])
        # Verification if > or < than bounds
        V[i,d] = de.boundary(V[i,d], lb[0,d], ub[0,d])
    
    # Random one dimension from 1 to dim
    index = np.random.randint(low = 0, high = dim)
    # crossover - rand/1/bin --- maybe use the multi crossover in: Multi‑variant differential evolution algorithm for feature selection
    # 0-20% ---> DE/rand/2 -- exploration
    # 20-40% --> DE/rand/1 -- exploration
    # 40-60% --> DE/current-to-best/1 -- balanced
    # 60-80% --> DE/best/2 -- exploitation
    # 80-100% -> DE/best/1 -- purely exploitation

    ### Differential Evolution Mutations: Taxonomy, Comparison and Convergence Analysis 
    for d in range(dim):
        #if (rand() <= CR)  or  (d == index):
        if (rand() <= crossover_rate_T)  or  (d == index):
            U[i,d] = V[i,d]
        else:
            U[i,d] = X[i,d]
    
    return U


def rand_2(i, N, V, U, X, lb, ub, dim, mutation_rate_T, crossover_rate_T):

    # Choose r1, r2, r3 randomly, but not equal to i 
    RN = np.random.permutation(N)
    for j in range(N):
        if RN[j] == i:
            RN = np.delete(RN, j)
            break

    r1 = RN[0]
    r2 = RN[1]
    r3 = RN[2]
    r4 = RN[3]
    r5 = RN[4]
    # mutation 
    for d in range(dim):
        #V[i,d] = X[r1,d] + F * (X[r2,d] - X[r3,d])
        V[i,d] = X[r1,d] + mutation_rate_T * (X[r2,d] - X[r3,d]) + mutation_rate_T * (X[r3,d] - X[r4,d])
        # Verification if > or < than bounds
        V[i,d] = de.boundary(V[i,d], lb[0,d], ub[0,d])
    
    # Random one dimension from 1 to dim
    index = np.random.randint(low = 0, high = dim)
    # crossover - rand/1/bin --- maybe use the multi crossover in: Multi‑variant differential evolution algorithm for feature selection
    # 0-20% ---> DE/rand/2 -- exploration
    # 20-40% --> DE/rand/1 -- exploration
    # 40-60% --> DE/current-to-best/1 -- balanced
    # 60-80% --> DE/best/2 -- exploitation
    # 80-100% -> DE/best/1 -- purely exploitation

    ### Differential Evolution Mutations: Taxonomy, Comparison and Convergence Analysis 
    for d in range(dim):
        #if (rand() <= CR)  or  (d == index):
        if (rand() <= crossover_rate_T)  or  (d == index):
            U[i,d] = V[i,d]
        else:
            U[i,d] = X[i,d]
    
    return U


def current_to_best(i, N, V, U, X, lb, ub, dim, mutation_rate_T, crossover_rate_T, best):

    # Choose r1, r2, r3 randomly, but not equal to i 
    RN = np.random.permutation(N)
    for j in range(N):
        if RN[j] == i:
            RN = np.delete(RN, j)
            break

    r1 = RN[0]
    r2 = RN[1]
    
    # mutation 
    for d in range(dim):
        #V[i,d] = X[r1,d] + F * (X[r2,d] - X[r3,d])
        V[i,d] = X[r1,d] + mutation_rate_T * (best[0, d] - X[i,d]) + mutation_rate_T * (X[r1,d] - X[r2,d])
        # Verification if > or < than bounds
        V[i,d] = de.boundary(V[i,d], lb[0,d], ub[0,d])
    
    # Random one dimension from 1 to dim
    index = np.random.randint(low = 0, high = dim)
    # crossover - rand/1/bin --- maybe use the multi crossover in: Multi‑variant differential evolution algorithm for feature selection
    # 0-20% ---> DE/rand/2 -- exploration
    # 20-40% --> DE/rand/1 -- exploration
    # 40-60% --> DE/current-to-best/1 -- balanced
    # 60-80% --> DE/best/2 -- exploitation
    # 80-100% -> DE/best/1 -- purely exploitation

    ### Differential Evolution Mutations: Taxonomy, Comparison and Convergence Analysis 
    for d in range(dim):
        #if (rand() <= CR)  or  (d == index):
        if (rand() <= crossover_rate_T)  or  (d == index):
            U[i,d] = V[i,d]
        else:
            U[i,d] = X[i,d]
    
    return U


def best_2(i, N, V, U, X, lb, ub, dim, mutation_rate_T, crossover_rate_T, best):

    # Choose r1, r2, r3 randomly, but not equal to i 
    RN = np.random.permutation(N)
    for j in range(N):
        if RN[j] == i:
            RN = np.delete(RN, j)
            break

    r1 = RN[0]
    r2 = RN[1]
    r3 = RN[2]
    r4 = RN[3]
    
    # mutation 
    for d in range(dim):
        #V[i,d] = X[r1,d] + F * (X[r2,d] - X[r3,d])
        V[i,d] = best[0, d] + mutation_rate_T * (X[r1, d] - X[r2,d]) + mutation_rate_T * (X[r3,d] - X[r4,d])
        # Verification if > or < than bounds
        V[i,d] = de.boundary(V[i,d], lb[0,d], ub[0,d])
    
    # Random one dimension from 1 to dim
    index = np.random.randint(low = 0, high = dim)
    # crossover - rand/1/bin --- maybe use the multi crossover in: Multi‑variant differential evolution algorithm for feature selection
    # 0-20% ---> DE/rand/2 -- exploration
    # 20-40% --> DE/rand/1 -- exploration
    # 40-60% --> DE/current-to-best/1 -- balanced
    # 60-80% --> DE/best/2 -- exploitation
    # 80-100% -> DE/best/1 -- purely exploitation

    ### Differential Evolution Mutations: Taxonomy, Comparison and Convergence Analysis 
    for d in range(dim):
        #if (rand() <= CR)  or  (d == index):
        if (rand() <= crossover_rate_T)  or  (d == index):
            U[i,d] = V[i,d]
        else:
            U[i,d] = X[i,d]
    
    return U

def best_1(i, N, V, U, X, lb, ub, dim, mutation_rate_T, crossover_rate_T, best):

    # Choose r1, r2, r3 randomly, but not equal to i 
    RN = np.random.permutation(N)
    for j in range(N):
        if RN[j] == i:
            RN = np.delete(RN, j)
            break

    r1 = RN[0]
    r2 = RN[1]
    
    # mutation 
    for d in range(dim):
        #V[i,d] = X[r1,d] + F * (X[r2,d] - X[r3,d])
        V[i,d] = best[0, d] + mutation_rate_T * (X[r1, d] - X[r2,d]) 
        # Verification if > or < than bounds
        V[i,d] = de.boundary(V[i,d], lb[0,d], ub[0,d])
    
    # Random one dimension from 1 to dim
    index = np.random.randint(low = 0, high = dim)
    # crossover - rand/1/bin --- maybe use the multi crossover in: Multi‑variant differential evolution algorithm for feature selection
    # 0-20% ---> DE/rand/2 -- exploration
    # 20-40% --> DE/rand/1 -- exploration
    # 40-60% --> DE/current-to-best/1 -- balanced
    # 60-80% --> DE/best/2 -- exploitation
    # 80-100% -> DE/best/1 -- purely exploitation

    ### Differential Evolution Mutations: Taxonomy, Comparison and Convergence Analysis 
    for d in range(dim):
        #if (rand() <= CR)  or  (d == index):
        if (rand() <= crossover_rate_T)  or  (d == index):
            U[i,d] = V[i,d]
        else:
            U[i,d] = X[i,d]
    
    return U
    
def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    return Xbin

def KNN3FOLD(run, X, Y, tipos):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits = 3)
    ####loo = LeaveOneOut()
    #param_grid = [{'n_neighbors': [3, 5, 7, 10, 15],'weights': ['uniform', 'distance']}]
    #model = GridSearchCV(KNeighborsClassifier(),param_grid,refit=True,verbose=0,n_jobs=nucleos, cv=kf)
    model = KNN(n_neighbors=10)
    #model = SVC(kernel='rbf',gamma='scale',C=5000, probability=False)
    #model = DT()
    #model = LR(solver='liblinear')   
    #arq = open(nome+"-KNN.csv","w")
    #arq.write("run,TN,FP,FN,TP,Accuracy,Precision,Recall,F1Score,Specificity,Lr+,Lr-\n")
    for i in tqdm(range(1,run)):  
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
        print(Accuracy)

    return Accuracy
        #arq.write(str(i)+","+str(round(TN,3))+","+str(round(FP,3))+","+str(round(FN,3))+","+str(round(TP,3))+","+str(round(Accuracy,3))+","+str(round(Precision,3))+","+str(round(Recall,3))+","+str(round(F1Score,3))+","+str(round(Specificity,3))+","+str(round(LRP,3))+","+str(round(LRM,3))+"\n")
    #arq.write("Média:"+str(round(mean(TN),3))+","+str(round(mean(FP),3))+","+str(round(mean(FN),3))+","+str(round(mean(TP),3))+","+str(round(mean(Accuracy),3))+","+str(round(mean(Precision),3))+","+str(round(mean(Recall),3))+","+str(round(mean(F1Score),3))+","+str(round(mean(Specificity),3))+","+str(round(mean(LRP),3))+","+str(round(mean(LRM),3)) )
    #arq.close()