import os
import argparse
import sys
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
import random
import pandas as pd
import matplotlib
from matplotlib import pyplot
import math
import time
from deap import base
from deap import creator
from deap import tools
import scipy.io as scio
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, classification_report
import itertools
import MOECA_utils as utils

def MOECA(W,max_gen,pop_size,num_obj,offspring_size,cross_rate,mutate_rate,ASD_reg,ADHD_reg,COBRE_reg):

    init_clu = np.zeros(190)
    dataset = [ASD_reg,ADHD_reg,COBRE_reg]

    evaluate = utils.FitnessEvaluation_regression(dataset, W)

    creator.create('FitnessMax', base.Fitness, weights=(1.0, 1.0, 1.0))
    creator.create('Individual', np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register('attr', (lambda init_gene: init_gene), init_clu)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr,n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate)

    # Initialize population
    pop2 = toolbox.population2(n=pop_size)
    pop = pop2

    # Fitness evaluation
    fits = toolbox.map(toolbox.evaluate, pop)

    i = 0
    start = time.time()
    for fit, ind in zip(fits, pop):
        print("{} pop".format(i)+ '-'*40)
        ind.fitness.values = fit
        print("ASD acc: ", ind.fitness.values[0])   
        print("ADHD acc: ", ind.fitness.values[1])  
        print("COBRE acc: ", ind.fitness.values[2])
        i = i + 1
    end = time.time()
    print("time :",end-start)

    # Evolution loop
    gen = 0

    print('Start evolution loop...')
    while gen < max_gen:
        print('Iteration {} gen'.format(gen))

        # Generation of offspring
        offspring = utils.reproduction(pop, offspring_size, W, toolbox,utils.sum_combanation, mutate_rate=mutate_rate, cross_rate=cross_rate)     
        
        for i, p in enumerate(pop):
            print('{}th pop'.format(i))
            print("ASD acc: ", p.fitness.values[0])   
            print("ADHD acc: ", p.fitness.values[1])  
            print("COBRE acc: ", p.fitness.values[2])
            
        srv = utils.SRV(pop,pop_size,num_obj)
        # Archive truncation
        pop = tools.selNSGA3(pop + offspring, k=pop_size, ref_points= srv) 
        gen += 1

    ASD_ACC, ADHD_ACC, COBRE_ACC = np.empty(pop_size), np.empty(pop_size), np.empty(pop_size)
    for i, p in enumerate(pop):
        ASD_ACC[i], ADHD_ACC[i],COBRE_ACC[i] = p.fitness.values[0], p.fitness.values[1],p.fitness.values[2]

