### Gabriel Dominico

### TODO
# verificar impacto do threshold para binariazição  --- Executar testes
# utilizar F1Score na avaliacao de fitness   --- DONE
# acrescentar linha do melhor/media individuo no grafico (F1Score) --- DONE
# testar versoes de crossover   --- DONE (Multi variant DE? Testar cada crossover separadamente?)
# aplicar filtering antes do DE (retirar dados com variancia 0 e correlação baixa)
# criar drive para compartilhar artigos/gerenciamento do projeto
# adicionar busca local para verificar impacto? SA? Hooke-Jeeves? Nelder-mead?
# implementar uma nova função de fitness (multi-objective maybe?) 
# 

### TODO TESTS
# quais datasets do CUMIDA utilizar? Executei todos para ter uma avaliação melhor (10 runs inicialmente)
# testar quais parametros? threshold? crossovers? threshold padrão = 0.5, testando com multi variant DE
# 

# testar algoritmo com algoritmos "básicos de feature selection"
# random forest feature selection

### IDEIAS
# 1 - Talvez eliminar indivíduos e gerar novos individuos considerando as features selecionadas pelo melhor individuo, ou seja,
# excluir individuo x e gerar um novo individuo com valores < 0.5 para features não selecionadas e o resto deixar de forma aleatoria
# irá diminuir a quantidade de features selecionadas e talvez melhorar o resultado.
###
# 2 - Criar um arquivo para salvar o melhor individuo e poder "resetar" a população seguindo a Ideia #1.
###
# 3 - Utilizar algum método para inicializar a população, talvez filter methods (https://academic.oup.com/jcde/article/9/3/949/6590608)
# e verificar o impacto. Outra ideia é utilizar a estratégia do SHADE (grande população no começo e ir diminuindo)
###


import os
import sys
import math
import time
import importlib
import statistics 
import pathlib
import csv

from de import DE    
import visualize
import RR_utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from pycm import *


### Classifiers worth using
from sklearn.svm import SVC                           ### SVC()                    --> SVM (Support Vector Machine)
from sklearn.tree import DecisionTreeClassifier as DT      ### DecisionTreeClassifier() --> DT
from sklearn.neural_network import MLPClassifier as MLP     ### MLPClassifier()          --> MLP
from sklearn.neighbors import KNeighborsClassifier as KNN   ### KNeighborsClassifier()   --> KNN
from sklearn.linear_model import LogisticRegression as LR  ### LogisticRegression()     --> LR
from sklearn.ensemble import RandomForestClassifier  as RF  ### RandomForestClassifier() --> RF

from sklearn.feature_selection import VarianceThreshold
import warnings


warnings.filterwarnings("ignore")


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# load data --- with header

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))
out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)
#print(out_file, out_fold)
#print(cfg.dataset_file)

file = cfg.dataset_file.split("/")

print(cfg.dataset_file)

temp_df = pd.read_csv("/home/gdominico/workspace/WrapperDE-FS-MO2/WrapperDE-FS/ARTIGO1/"+file[1]+"/selectors_silhouette_tsne2d.csv")
index = (temp_df.loc[temp_df['Weighted silhouette'].idxmax()])
#print(index)

#score_df = pd.read_csv("/home/dominico/Desktop/WrapperDE-FS/"+index[1], index_col=0)
score_df = pd.read_csv(index[1], index_col=0)
coin_vector = score_df['value']
#print(coin_vector)
#coin_vector_sum = coin_vector.sum()
#coin_vector_mean = coin_vector.mean()

df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

run = cfg.runs

C =  df[cfg.class_label].nunique()
tipos = df[cfg.class_label].unique()
D = len(df.columns) - 1
N = len(df.index)

#print("Original dataset: {}\n".format(cfg.dataset_file))
print('\nClasses:')
print(np.sort(df[cfg.class_label].unique()))
   
feat, label = RR_utils.get_XY2(df, cfg.task, cfg.class_label)
print(feat)
print(label)

### split data into train & validation (70 -- 30) /// maybe LOO?
#xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)

#fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# acc_knn = RR_utils.KNN3FOLD(100, feat, label, tipos)
# print(acc_knn)

# sys.exit()


number_features_selected = []

results = {
    "media_erro" : [],
    "std_erro" : [],
    "media_features" : [],
    "std_features" : [],
    "best" : [],
    "best_individual" : []
}   

for i in tqdm(range(1,run)):
    ### initialization of variables
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
    Support = 0  
    cnf_matrix_aux = np.zeros((int(C),int(C)))
    #cnf_matrix_aux_pycm = np.zeros((int(C),int(C)))

    # parameters of DE -- maybe use jDE for CR and F -- using jDE for self-adaptive parameters
    thres= 0.6
    k    = 5     # neighbors in KNN
    N    = 40   # number of individuals 
    T    = 200   # maximum number of iterations
    lb   = 0     # lower bound for individuals
    ub   = 1.0     # upper bound for individuals

    #opts = {'threshold':thres, 'k':k, 'fold':fold, 'N':N, 'T':T, 'lb':lb, 'ub':ub, 'Classes':C, 'feat':feat, 'label': label, 'out_fold': out_fold, 'df': df, 'cfg': cfg, 'coin_vector': coin_vector}
    opts = {'threshold':thres, 'k':k, 'N':N, 'T':T, 'lb':lb, 'ub':ub, 'Classes':C, 'feat':feat, 'label': label, 'out_fold': out_fold, 'df': df, 'cfg': cfg, 'coin_vector': coin_vector}

    ### Feature selection using DE
    fmdl, results = DE(feat, label, opts, cfg, i, results)
    sf   = fmdl['sf']

    number_features_selected.append(fmdl['nf'])

    ### Using the results found by the DE, now we use the selected features to train/test the classifier
    ### sf = vector with the index of which feature was selected by the DE algorithm.

#     X = feat[:, sf]
#     Y = label
#     kf = StratifiedKFold(n_splits = 10)
#     #loo = LeaveOneOut()

#     #model = KNeighborsClassifier(n_neighbors = k)
#     #model = LogisticRegression()   
#     #model = SVC()
#     model = DT()
#     count = 0
#     for train_indices, test_indices in kf.split(X, Y):   ### stratified k-fold
#     #for train_indices, test_indices in loo.split(X):    ### leave one out 
        
#         train_X, train_y = X[train_indices], Y[train_indices]
#         #print(train_X, train_y)
#         test_X, test_y = X[test_indices], Y[test_indices]
#         #print(test_X, test_y)
#         pred=model.fit(train_X, train_y).predict(test_X)
#         #tn, fp, fn, tp  = confusion_matrix(test_y, pred, labels=np.sort(df[cfg.class_label].unique())).ravel()
#         cnf_matrix  = confusion_matrix(test_y, pred, labels=np.sort(df[cfg.class_label].unique()))
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
#     cnf_matrix_aux_pycm.save_html(os.path.join(str(pathlib.Path().resolve())+"/"+str(out_fold),"confusion_matrix"+str(i)))
#     #print(os.path.join(str(pathlib.Path().resolve())+"/"+str(out_fold),"confusion_matrix"+str(i)))
#     #arq.write(cnf_matrix_aux_pycm)
#     arq.write(str(i)+"  ,  "+str("{:3f}".format(Accuracy))+"   ,  "+str("{:3f}".format(Precision))+"  ,  "+str("{:3f}".format(Recall))+"  ,  "+str("{:3f}".format(F1Score))+"  ,  "+str("{:3f}".format(Specificity))+"     ,  "+str("{:3f}".format(LRP))+"  ,  "+str("{:3f}".format(LRM))+"  ,  "+str((number_features_selected[i-1]))+"\n")
# arq.close()

# Number of selected features

#print(results)
#teste = []
results['media_erro'].append(sum(results["media_erro"])/len(results["media_erro"]))
results['std_erro'].append(sum(results["std_erro"])/len(results["std_erro"]))
results['media_features'].append(sum(results["media_features"])/len(results["media_features"]))
results['std_features'].append(sum(results["std_features"])/len(results["std_features"]))
results['best'].append(max(results["best"]))
results['best_individual'].append("NADA")


#print(sum(results["media"])/len(results["media"]))
#print(sum(results["std"])/len(results["std"]))

with open(out_fold+"/results.csv", "w") as outfile:
 
    # pass the csv file to csv.writer function.
    writer = csv.writer(outfile)
 
    # pass the dictionary keys to writerow
    # function to frame the columns of the csv file
    writer.writerow(results.keys())
   
    # make use of writerows function to append
    # the remaining values to the corresponding
    # columns using zip function.
    writer.writerows(zip(*results.values()))



num_feat = fmdl['nf']
#if run > 2:
    #print(number_features_selected)
    #print("Mean number of features selected: %s +/- %s" % (statistics.mean(number_features_selected), statistics.stdev(number_features_selected)))

#print("features selected:"+str(sum(results["media"])/len(results["media"]))+" +/- "+str(sum(results["std"])/len(results["std"])))




#visualize.plot_convergence(fmdl, opts, out_fold)

#visualize.stacked_plot(fmdl, opts, out_fold)



