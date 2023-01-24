#[1997]-"Differential evolution - A simple and efficient heuristic for global optimization over continuous spaces"

### Workflow of the algorithm:
### Initial population with the specified parameters
### Binary conversion for initial population fitness avaliation
### Print best solution within the population and store for plotting purposes
### DE evolutive cycle:
###     1) Selection of random indexes differente from current individual
###     2) Mutation --- for now using 5 different routines
###     3) Crossover 
### Binary conversion and fitness avaliation of generated individual
### Selection of the individual --- same scheme used in canonical DE (better = stay/enter population -- elitism)
### 

import numpy as np
from numpy.random import rand
import function
import random
import RR_utils
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from paretoset import paretoset, paretorank
import pandas as pd
import csv

from pycm import *
import os
import pathlib
import visualize

### Generation of initial population, here we will generate each individual between [0,1) for the crossover/mutation operations of the DE
def init_population(lb, ub, N, dim, thres, coin_vector):
    coin_vector_mean = coin_vector.mean()
    X = np.zeros([N, dim], dtype='float')
    #media = (np.mean(clf.feature_importances_))
    #arr = clf.feature_importances_
    for i in range(N):
        for d in range(dim):
            if i <= 10:
                if coin_vector[d] > coin_vector_mean:
                    X[i,d] = thres + ((ub[0,d] - thres) * rand())
#                print(X[i,d])
            #    #X[i,d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
                else:
                    X[i,d] = thres * rand()        
            elif i > 10 and i <= 30:
                X[i,d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
            else: 
                X[i,d] = lb[0, d] + 0.5 + (ub[0, d] - lb[0, d]) * rand()
                #print(X[i,d])
        #print(np.count_nonzero(X[i]>0.6))
    #print(X)
    return X

### Binary conversion using the threshold method (naive but effective --- Paper: Wrapper-based feature selection via differential
                                                                          #evolution: benchmarking different discretisation techniques)
# def binary_conversion(X, thres, N, dim):
#     Xbin = np.zeros([N, dim], dtype='int')
#     for i in range(N):
#         for d in range(dim):
#             if X[i,d] > thres:
#                 Xbin[i,d] = 1
#             else:
#                 Xbin[i,d] = 0
#     return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x

def updatejDE(mutation_rate_T, crossover_rate_T, mutation_rate, crossover_rate, N):
    Fl = 0.1
    Fu = 0.9
    tau1 = tau2 = 0.1


    rand1 = random.uniform(0, 1)
    rand2 = random.uniform(0, 1)
    rand3 = random.uniform(0, 1)
    rand4 = random.uniform(0, 1)

    for ind in range(N):
        if rand2 < tau1:
            mutation_rate_T[ind] = Fl + (rand1 * Fu)
        else:                   
            mutation_rate_T[ind] = mutation_rate[ind]

        if rand4 < tau2:
            crossover_rate_T[ind] = rand3
        else:
            crossover_rate_T[ind] = crossover_rate[ind]
    return mutation_rate_T, crossover_rate_T


def DE(xtrain, ytrain, opts, cfg, run, results):

    # Parameters
    CR    = 0.9     # crossover rate
    F     = 0.5     # factor
    
    thres    = opts['threshold']
    N        = opts['N']
    max_iter = opts['T']

    #clf = opts['clf']
    out_fold = opts['out_fold']
    df = opts['df']
    coin_vector = opts['coin_vector']

    if 'CR' in opts:
        CR   = opts['CR'] 
    if 'F' in opts:
        F    = opts['F']
    if 'lb' in opts:
        lb   = opts['lb']
    if 'ub' in opts:
        ub   = opts['ub']

    statistics = {
        "TP" : [],
        "FN" : [],
        "FP" : [],
        "TN" : [],
        "Accuracy" : [],
        "Precision" : [],
        "Recall" : [],
        "F1Score" : [],
        "Specificity" : [],
        "LRP" : [],
        "LRM" : [],
        "pop" : []
    }


    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')
    # crossover and mutation rate for each individual (jDE)
    crossover_rate_T  = [random.gauss(0.5, 0.1) for i in range(N)] #[0.9] * N  ### Initial crossover ratio for jDE  
    mutation_rate_T   = [0.5] * N  ### Initial mutation rate for jDE
    crossover_rate    = [random.gauss(0.5, 0.1) for i in range(N)]  ### 
    mutation_rate     = [0.5] * N  ### 
            
    # Initialize population
    X     = init_population(lb, ub, N, dim, thres, coin_vector)
    # Binary conversion
    Xbin  = RR_utils.binary_conversion(X, thres, N, dim)

    #print(Xbin)
    
    # Initialization of variables
    fit   = np.zeros([N, 1], dtype='float')
    fit_features   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = 0
    fit_featuresG = float('inf')
    multiobjective_fitness = []
    # Fitness at first iteration
    for i in range(N):
        fit[i,0], fit_features[i, 0], cnf_matrix_pycm, pareto_parent = function.fitness(xtrain, ytrain, Xbin[i,:], opts, out_fold, 0, df, cfg, run, multiobjective_fitness, statistics)
        if fit[i,0] > fitG and fit_features[i,0] < fit_featuresG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]
            fit_featuresG = fit_features[i,0]
            fitF1 = fit[i,0] #fit_features[i,0]
    
    # initialization of list of best individual for plotting
    curve = np.zeros([1, max_iter], dtype='float') 
    curvef1score = np.zeros([1, max_iter], dtype='float') 
    curve_mean = np.zeros([1, max_iter], dtype='float') 
    t     = 0
    
    curve[0,t] = fitG.copy()
    curvef1score[0,t] = fitF1.copy()
    #print("Generation:", t + 1)
    #print("Best solution found:", curve[0,t])
    #time.sleep(10)
    t += 1

    pos_aux = np.asarray(range(0, dim)) 
    feat_index = []
    new_genes = []
    total_genes = []

    flag_best = 0

    while t < max_iter:  
        #print("Generation:", t + 1)
        V = np.zeros([N, dim], dtype='float')
        U = np.zeros([N, dim], dtype='float')

        mutation_rate_T, crossover_rate_T = updatejDE(mutation_rate_T, crossover_rate_T, mutation_rate, crossover_rate, N)

        for i in range(N):    

            # Crossover/mutation evolutive cycle
            #if t <= max_iter*0.2:
            U = RR_utils.rand_1(i, N, V, U, X, lb, ub, dim, mutation_rate_T[i], crossover_rate_T[i])    
            # elif t > max_iter*0.2 and t <= max_iter*0.4:
            #     U = RR_utils.rand_2(i, N, V, U, X, lb, ub, dim, mutation_rate_T[i], crossover_rate_T[i])    
            # elif t > max_iter*0.4 and t <= max_iter*0.6:
            #     U = RR_utils.current_to_best(i, N, V, U, X, lb, ub, dim, mutation_rate_T[i], crossover_rate_T[i], Xgb)    
            # elif t > max_iter*0.6 and t <= max_iter*0.8:
            #     U = RR_utils.best_2(i, N, V, U, X, lb, ub, dim, mutation_rate_T[i], crossover_rate_T[i], Xgb)
            # elif t >= max_iter*0.8:
            #     U = RR_utils.best_1(i, N, V, U, X, lb, ub, dim, mutation_rate_T[i], crossover_rate_T[i], Xgb)
            
            
        # Binary conversion
        Ubin = RR_utils.binary_conversion(U, thres, N, dim)
        multiobjective_fitness = []
        # Selection
        #print("POPULAÇÃO GERAÇÃO ", t)
        #RR_utils.print_binary_population(X, N)
        for i in range(N):
            
            pareto_test = []
            fitU, fit_featureU, cnf_matrix_pycm, pareto = function.fitness(xtrain, ytrain, Ubin[i,:], opts, out_fold, t, df, cfg, run, multiobjective_fitness, statistics)
            #print(pareto)
            #print(pareto[i], pareto_parent[i])        #pareto = new individual, pareto_parent = old individual (parent)
            pareto_test.append(pareto_parent[i])
            pareto_test.append(pareto[i])
            mask = paretoset(pareto_test, sense=["max", "min"])
            cost = paretorank(pareto_test, sense=["max", "min"])
            #print(mask)
            #print(cost)
            #time.sleep(10)

            #print(fitF1value, fitU)

            if cost[1] < cost[0]:
                X[i,:]   = U[i,:]
                fit[i,0] = fitU
                fit_features[i,0] = fit_featureU
                mutation_rate[i] = mutation_rate_T[i]
                crossover_rate[i] = crossover_rate_T[i]
                pareto_parent[i] = pareto[i]

            # if fitU <= fit[i,0]:
            #     X[i,:]   = U[i,:]
            #     fit[i,0] = fitU
            #     fit_features[i,0] = fitF1value
            #     mutation_rate[i] = mutation_rate_T[i]
            #     crossover_rate[i] = crossover_rate_T[i]
            if fit[i,0] > fitG:
                #print("Best anterior", fitG, fit_featuresG)
                #print("Best atual", fit[i,0], fit_features[i,0])
                flag_best = 1
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]   
                fitF1 = fit[i,0]
                fit_featuresG = fit_features[i,0]
                best_matrix = cnf_matrix_pycm

    #    if cfg.debug == 1:

        Gbin       = RR_utils.binary_conversion(Xgb, thres, 1, dim) 
        Gbin       = Gbin.reshape(dim)
        curve[0,t] = fitG.copy()
        curvef1score[0,t] = fitF1.copy()
        curve_mean[0,t] = fit.mean()
        #print(fit.mean())
        #print(fit[0,t])
        #print(curve[0,t], curvef1score[0,t])
        #curveF1Score[0, t] = fitGScore1.copy()
        pos        = np.asarray(range(0, dim))            
        sel_index  = pos[Gbin == 1]
        total_genes.append(len(sel_index))
        num_feat   = len(sel_index)
        #     #print("Generation:", t + 1)
        #     #print("Best solution found (vector):", Xgb)
        #     #print("Best solution found (binary vector):", Gbin)
        #     print("Best solution found (fitness):", curve[0,t])
        #     #print("Number of features selected:", num_feat)
        t += 1            
        # else:
        #     curve[0,t] = fitG.copy()
        #     print("Best solution found (fitness):", curve[0,t])
        #     t += 1            

        #print(curve, curveF1Score)

        j = 0
        
        for i in range(len(sel_index)):            
            if sel_index[i] not in feat_index:                
                feat_index.append(sel_index[i])
                j += 1
        new_genes.append(j)

    if flag_best == 0:
        best_matrix = cnf_matrix_pycm
    best_matrix.save_html(os.path.join(str(pathlib.Path().resolve())+"/"+str(out_fold)+"runs/","confusion_matrix"+"_"+str(run)))

    paretoglobal = []

    #flat_list = (np.concatenate(1-fit).flat)
    flat_list = (np.concatenate(fit).flat)
    flat_list_feature = (np.concatenate(np.round_(fit_features*len(Ubin[0]))).flat)

    #print(fit_features.mean()*dim)

    df = pd.DataFrame({'x':flat_list, 'y':flat_list_feature})



    mask_global = paretoset(df, sense=["max","min"])
    top_individuals = df[mask_global]  

    #print(top_individuals)      

    # Best feature subset
    Gbin       = RR_utils.binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    #print("Best solution found (binary vector):", Xgb)
    #print("Best solution after binary conversion", Gbin, fitG)

    # Create dictionary
    #diversity = RR_utils.get_diversity(len(sel_index), X, N)            
    #print("Diversity:", diversity)
    #Ubin = binary_conversion(U, thres, N, dim)
    #RR_utils.print_binary_population(Ubin, N)
    de_data = {'sf': sel_index, 'c': curve, 'nf': num_feat, 'new_genes': new_genes, 'total_genes': total_genes, 'f1': curvef1score, 'mean': curve_mean}

    #print(X)
    #Xbin_final = []
    #for i in range(0, len(X)):
    #Xbin_final = binary_conversion(X, thres, N, dim)
    #Xbin_final.append(binary_conversion(X, thres, N, dim))
    #print(fit_features.mean()*dim, fit_features.std()*dim)    
    #print(fit.mean())
    #print(fit.std())

    results["best"].append(fit.max())
    results["best_individual"].append(Gbin)

    results["media_erro"].append(fit.mean())
    results["std_erro"].append(fit.std())

    results["media_features"].append(fit_features.mean()*dim)
    results["std_features"].append(fit_features.std()*dim)

    #print(Ubin.mean()*dim, Ubin.std())
    #print(Xbin_final)
    #Xbin_final.append()

    #statistics["pop"].append(binary_conversion(X, thres, N, dim))

    with open(out_fold+"runs/statistics_"+str(run)+".csv", "w") as outfile: 
        # pass the csv file to csv.writer function.
        writer = csv.writer(outfile)     
        # pass the dictionary keys to writerow
        # function to frame the columns of the csv file
        writer.writerow(statistics.keys())       
        # make use of writerows function to append
        # the remaining values to the corresponding
        # columns using zip function.
        writer.writerows(zip(*statistics.values()))


    visualize.stacked_plot(de_data, opts, out_fold+"runs/", run)
    visualize.plot_pareto(df, top_individuals, len(Ubin[0]), out_fold+"runs/", run)
    
    return de_data, results