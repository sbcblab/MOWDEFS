# Bruno Iochins Grisci
# June 23rd, 2022

import os
import sys
import numpy as np ; na = np.newaxis
from scipy.stats import gmean
import pandas as pd
import importlib
import importlib.util
from collections import namedtuple
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier       ### DecisionTreeClassifier() --> DT
from sklearn.neural_network import MLPClassifier      ### MLPClassifier()          --> MLP
from sklearn.neighbors import KNeighborsClassifier    ### KNeighborsClassifier()   --> KNN
from sklearn.linear_model import LogisticRegression   ### LogisticRegression()     --> LR
from sklearn.ensemble import RandomForestClassifier   ### RandomForestClassifier() --> RF
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import metrics


import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

SCORE_LABEL = 'score'

##########################################################################################


if __name__ == '__main__':
    out_fold, out_file = RR_utils.create_output_dir(cfg.dataset_file, cfg.output_folder)

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    if cfg.cv_splits is None:
        splits = RR_utils.split_cv(df, task=cfg.task, class_label=cfg.class_label, k=cfg.k)
        np.set_printoptions(threshold=sys.maxsize)
        with open(out_fold+"split.py", "w") as sf:
            sf.write("from collections import namedtuple\nfrom numpy import array\nSplit = namedtuple('Split', ['tr', 'te'])\nsplits = {}".format(splits))
    if cfg.cv_splits is not None:
        spt_file = cfg.cv_splits
    else:    
        spt_file = '{}split.py'.format(out_fold)
    print(spt_file)
    #spt = importlib.import_module(spt_file.replace('/','.').replace('.py',''))

    spec = importlib.util.spec_from_file_location(spt_file.replace('/','.').replace('.py',''), spt_file) 
    spt  = importlib.util.module_from_spec(spec) 
    spec.loader.exec_module(spt) 

    splits = spt.splits

    if not os.path.exists(out_fold+'rf_eval/'):
        os.makedirs(out_fold+'rf_eval/')

    df = pd.read_csv(cfg.dataset_file, delimiter=cfg.dataset_sep, header=0, index_col=cfg.row_index)
    df = RR_utils.check_dataframe(df, cfg.class_label, cfg.task)

    sort_class = []
    if cfg.task == 'classification':
        sort_class = np.sort(df[cfg.class_label].unique())
    elif cfg.task == 'regression':
        sort_class, target_classes = RR_utils.split_targets(df, df[cfg.class_label].astype(float).min(), df[cfg.class_label].astype(float).max(), cfg.target_split, cfg.class_label)
    print(sort_class)

    if cfg.task == 'regression':
        df[cfg.class_label] = RR_utils.shift_target(df, cfg.class_label)

    ranks_values = []
    ranks_labels = []

    #for fold in range(len(splits)):
    for fold in range(1):
        print('\n###### {}-FOLD:\n'.format(fold+1))
        #out = pd.read_csv('{}_{}_out.csv'.format(out_file, fold+1), delimiter=',', header=0, index_col=0)
        #load a neural network

        tr_df, te_df, mean_vals, std_vals, min_vals, max_vals = RR_utils.split_data(df, splits[fold], cfg.class_label, cfg.standardized, cfg.rescaled) 
        tr_df2, te_df2, mean_vals2, std_vals2, min_vals2, max_vals2 = RR_utils.split_data(df, splits[fold], cfg.class_label, False, False) 

        if te_df.empty:
            l = [(tr_df, 'train', tr_df2)]
        else:
            l = [(tr_df, 'train', tr_df2), (te_df, 'test', te_df2)]

        acc_vector = []
        for dataset in l:
            #print('### {}:'.format(dataset[1]))
            #print(dataset[0])
            X, Y = RR_utils.get_XY2(dataset[0], cfg.task, cfg.class_label)
            #print(X, Y)
            for i in range(0,5):
                #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test
                clf = SVC()
                scores = cross_val_score(clf, X, Y, cv=10)
                #clf = clf.fit(X_train,y_train)
                #y_pred = clf.predict(X_test)
                acc_vector.append(scores.mean())


                #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            print(np.mean(acc_vector))

            #print(clf.feature_importances_)
            #print(clf.feature_importances_.shape)
            #new_labels = list(dataset[0].columns)
            #new_labels.remove(cfg.class_label)
            #cdf = pd.DataFrame(data=clf.feature_importances_, index=new_labels, columns=['value'])
            #print(cdf)
            #class_col = 'score'
            #cdf.to_csv(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + 'DecisionTree_' + class_col + '_' + os.path.basename(cfg.dataset_file))

            