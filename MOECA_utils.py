import numpy as np
import numba
from sklearn import *
from sklearn.metrics import calinski_harabasz_score


def cluster_label(clu_cen,W):
    label = np.zeros(W)
    for i, cluster in enumerate(clu_cen):
        for j,element in enumerate(cluster):
            label[element] = i
    return label

def CH_score(data,label):
    avg = np.mean(data,axis=0)
    score = calinski_harabasz_score(avg,label)
    return score


class FitnessEvaluation(object):
    def __init__(self, dataset, W):
        super(FitnessEvaluation, self).__init__()
        [ASD,ADHD,COBRE,TC] = dataset
        self.ASD = ASD
        self.ADHD = ADHD
        self.COBRE = COBRE
        self.TC = TC
        self.W = W

        
    def __call__(self, pop):
            clu_cen = pop
            print("# of cluster group is "+ str(len(clu_cen)))

            label = cluster_label(clu_cen,self.W)
            CH_ASD = CH_score(self.ASD,label)
            CH_ADHD = CH_score(self.ADHD,label)
            CH_COBRE = CH_score(self.COBRE,label)
            CH_TC = CH_score(self.TC,label)

            return CH_ASD, CH_ADHD, CH_COBRE, CH_TC

class FitnessEvaluation_regression(object):
    def __init__(self, dataset, W):
        super(FitnessEvaluation_regression, self).__init__()
        [ASD_reg,ADHD_reg,COBRE_reg] = dataset
        self.W = W
        self.ASD_reg = ASD_reg
        self.ADHD_reg = ADHD_reg
        self.COBRE_reg = COBRE_reg

    def __call__(self, pop):

        clu_cen = pop
        print("# of cluster group is "+ str(len(clu_cen)))
        
        ASD_acc = self.ASD_reg .predict(pop)
        ADHD_acc = self.ADHD_reg.predict(pop)
        COBRE_acc = self.COBRE_reg.predict(pop)
        return ASD_acc[0], ADHD_acc[0], COBRE_acc[0]


@numba.jit
def mutation(ind, width, toolbox,sum_combanation, mutate_rate=0.0001):
    new_ind = toolbox.clone(ind)
    mask = np.random.rand(ind.size) < mutate_rate #replace
    for i in np.where(mask)[0]:
        r = np.random.randint(len(sum_combanation))
        new_ind[0][i] = sum_combanation[r] 

    return new_ind


@numba.jit
def crossover(ind1, ind2, toolbox, cross_rate=0.7):
    new_ind1, new_ind2 = toolbox.clone(ind1), toolbox.clone(ind2)

    if np.random.rand() > cross_rate:
        return new_ind1, new_ind2

    mask = np.random.rand(ind1.size) < 0.5
    new_ind1[0][mask] = ind2[0][mask]
    new_ind2[0][mask] = ind1[0][mask]
    return new_ind1, new_ind2

@numba.jit
def reproduction(pop, offspring_size, width, toolbox, sum_combanation, mutate_rate=0.05, cross_rate=0.7):
    offspring = []
    pop_size = len(pop)
    while len(offspring) < offspring_size:
        # Random selection
        print("Random selection")
        c = np.random.choice(np.arange(pop_size), 2, replace=False)
        # Crossover and mutation
        print("Crossover and mutation")
        child1, child2 = crossover(pop[c[0]], pop[c[1]], toolbox, cross_rate)
        child1 = mutation(child1, width, toolbox, sum_combanation, mutate_rate=mutate_rate)
        child2 = mutation(child2, width, toolbox, sum_combanation, mutate_rate=mutate_rate)
        # Fitness evaluation
        print("Fitness evaluation")
        child1.fitness.values = toolbox.evaluate(child1)
        child2.fitness.values = toolbox.evaluate(child2)
        # Resampling if constraint is violated
        if child1.fitness.values[0] != np.inf:
            offspring.append(child1)
        if child2.fitness.values[0] != np.inf:
            offspring.append(child2)
    return offspring[:offspring_size]