import os
import itertools
from collections import namedtuple
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import to_rgba


def plot_convergence(fmdl, opts, out_fold):

# Plot convergence of DE (Best results)

    curve   = fmdl['c']
    curve   = curve.reshape(np.size(curve,1))
    x       = np.arange(0, opts['T'], 1.0) + 1.0

    fig, ax = plt.subplots()
    ax.plot(x, curve, 'o-')
    ax.set_xlabel('Number of Iterations')
    ax.set_ylabel('Fitness')
    ax.set_title('DE')
    ax.grid()
    plt.savefig(out_fold + 'convergence_plot.pdf')  
    plt.show()
    

def stacked_plot(fmdl, opts, out_fold, run):

    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1, figsize=(16, 12))
    
    ax.set_axisbelow(True)  
    ax.yaxis.grid(color='gray', linestyle='solid', alpha=0.3)
    ax.xaxis.grid(color='gray', linestyle='solid', alpha=0.3)
    


    data = []
    curve   = fmdl['new_genes']
    total   = fmdl['total_genes']
    
    data.append(curve[0])

    curveFitness = fmdl['f1']
    curveFitness   = curveFitness.reshape(np.size(curveFitness,1))
    curveFitness   = curveFitness[1:opts['T']]

    curveFitness_mean = fmdl['mean']
    curveFitness_mean   = curveFitness_mean.reshape(np.size(curveFitness_mean,1))
    curveFitness_mean   = curveFitness_mean[1:opts['T']]


    #curveFitness   = 1-curveFitness

    #print(curveF1Score)

    #for i in range(len(curve)-1):
    #   print(curve[i+1], curve[i])
    #   data.append(abs(curve[i+1] - curve[i]))

    
    
    data = np.asarray(curve)
    total = np.asarray(total)
    #print(data)
    #print(total)
    #print(total-data)

    A = total-data
    B = total-data
    B[0] = data[0]
    for i in range(1, len(A)-1):
        B[i] = B[i-1] + data[i]
    
    x       = np.arange(1, opts['T'], 1.0) + 1.0
    plt.stackplot(x, B, color='#D9F6FC', alpha = 0.6)
    #plt.fill_between(x,B, color = '#D9F6FC', alpha=0.3)
    plt.bar(x, A, color='#77b5fe')
    plt.bar(x, data, bottom=A, color='#99EBA8')
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of selected features from best individual")
    plt.legend(["Total visited features", "Old features", "New features"], loc=0)
    plt.title("MOWDEFS")
    ax2 = ax.twinx()
    #ax2.scatter(x, curveFitness, color='r')
    ax2.plot(x, curveFitness,  color='r', linestyle='dashed')
    ax2.plot(x, curveFitness_mean,  color='b', linestyle='dashed')     
    ax2.legend(["Best Accuracy", "Mean Accuracy"], loc=2)
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.1)
    plt.savefig(out_fold + 'gene_selection_plot_dashed_'+str(run)+'.pdf')  
    #plt.show()

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


    #curve   = curve.reshape(np.size(curve,1))
    #x       = np.arange(0, opts['T']-1, 1.0) + 1.0

    #fig, ax = plt.subplots()
    #ax.plot(x, data, 'o-')
    #ax.set_xlabel('Number of Iterations')
    #ax.set_ylabel('Fitness')
    #ax.set_title('DE')
    #ax.grid()
    #plt.show() 

def plot_pareto(df, top_individuals, max_feature, out_fold, run):

    plt.figure(figsize=(8, 8), dpi=80)
    plt.title("Solutions in the Pareto set")
    plt.scatter(df["x"], df["y"], zorder=10, label="All solutions", s=30, alpha=0.8)

    #rint(df["x"].mean(), df["x"].std())
    #print(df["y"].mean(), df["y"].std())

    plt.scatter(
        top_individuals["x"],
        top_individuals["y"],
        zorder=5,
        label="Top solutions",
        s=200,
        alpha=1,
        marker="*",
    )

    plt.legend()
    plt.xlim([df["x"].min()-0.05, min(df["x"].max()+0.05, 1.0)])
    plt.ylim([df["y"].min()-0.1, df["y"].max()+0.1])
    plt.title("MOWDEFS Pareto set")
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Features")
    plt.grid(True, alpha=0.5, ls="--", zorder=0)
    plt.tight_layout()
    plt.savefig(out_fold + 'pareto_frontier_'+str(run)+'.pdf')  

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()