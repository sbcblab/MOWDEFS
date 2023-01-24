# Bruno Iochins Grisci
# January 26th, 2022

import os
import sys
import numpy as np ; na = np.newaxis
from scipy.stats import gmean
import pandas as pd
import importlib
import importlib.util
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler    
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from collections import namedtuple
from sklearn import linear_model

import RR_utils

config_file = sys.argv[1]
cfg = importlib.import_module(config_file.replace('/','.').replace('.py',''))

SCORE_LABEL = 'score'

# #########################################################################################


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

    if not os.path.exists(out_fold+'lasso_eval/'):
        os.makedirs(out_fold+'lasso_eval/')

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

        for dataset in l:
            print('### {}:'.format(dataset[1]))
            print(dataset[0])
            X, Y = RR_utils.get_XY(dataset[0], cfg.task, cfg.class_label)
            pipeline = Pipeline([('scaler',StandardScaler()),('model',Lasso())])
            #model = Lasso()
            clf = GridSearchCV(pipeline,{'model__alpha':np.arange(0.09,0.1,0.01)},cv = 5, scoring="neg_mean_squared_error",verbose=4)
            #clf = Lasso(alpha=5000,positive=True,fit_intercept=False,max_iter=1000,tol=0.0001)
            clf.fit(X,Y)
            #clf.fit(X, Y)
            #print(clf.best_params_)
            #print(clf.coef_[0])
            #print(clf.intercept_)
            #import eli5
            #eli5.show_weights(clf, top=-1, feature_names = X.columns.tolist())
            coefficients = clf.best_estimator_.named_steps['model'].coef_
            importance = np.abs(coefficients)
            print(importance)

            new_labels = list(dataset[0].columns)
            new_labels.remove(cfg.class_label)
            #print(new_labels)
            cdf = pd.DataFrame(data=importance[0], index=new_labels, columns=['value'])
            print(cdf)
            class_col = 'score'
            cdf.to_csv(cfg.output_folder + '/' + os.path.basename(cfg.dataset_file).replace('.csv','/') + 'Lasso_' + class_col + '_' + os.path.basename(cfg.dataset_file))

            
