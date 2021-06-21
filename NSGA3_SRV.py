import copy
import numpy as np
from deap import tools
import math

class ReferencePoint(list):
    '''A reference point exists in objective space an has a set of individuals
    associated to it.'''
    def __init__(self, *args):
        list.__init__(self, *args)
        self.associations_count = 0
        self.associations = []

def generate_reference_points(num_objs, num_divisions_per_obj=4):
    '''Generates reference points for NSGA-III selection.'''
    def gen_refs_recursive(work_point, num_objs, left, total, depth):
        if depth == num_objs - 1:
            work_point[depth] = left/total
            ref = ReferencePoint(copy.deepcopy(work_point))
            return [ref]
        else:
            res = []
            for i in range(left):
                work_point[depth] = i/total
                res = res + gen_refs_recursive(work_point, num_objs, left-i, total, depth+1)
            return res
    return gen_refs_recursive([0]*num_objs, num_objs, num_objs*num_divisions_per_obj,
                              num_objs*num_divisions_per_obj, 0)

def find_ideal_point(individuals):
    'Finds the ideal point from a set individuals.'
    current_ideal = [np.infty] * len(individuals[0].fitness.values)
    for ind in individuals:
        # Use wvalues to accomodate for maximization and minimization problems.
        current_ideal = np.minimum(current_ideal, 
                                  np.multiply(ind.fitness.values, 1))
    return current_ideal

def find_extreme_points(individuals):
    'Finds the individuals with extreme values for each objective function.'
    return [sorted(individuals, key=lambda ind:ind.fitness.values[o] * 1)[-1]
            for o in range(len(individuals[0].fitness.values))]

def construct_hyperplane(individuals, extreme_points):
    'Calculates the axis intersects for a set of individuals and its extremes.'
    def has_duplicate_individuals(individuals):
        for i in range(len(individuals)):
            for j in range(i+1, len(individuals)):
                if individuals[i].fitness.values == individuals[j].fitness.values:
                    return True
        return False

    num_objs = len(individuals[0].fitness.values)

    if has_duplicate_individuals(extreme_points):
        intercepts = [extreme_points[m].fitness.values[m] for m in range(num_objs)]
    else:
        b = np.ones(num_objs)
        A = [point.fitness.values for point in extreme_points]
        x = np.linalg.solve(A,b)
        intercepts = 1/x
    return intercepts

def normalize_objective(individual, m, intercepts, ideal_point, epsilon=1e-20):
    'Normalizes an objective.'
    # Numeric trick present in JMetal implementation.
    if np.abs(intercepts[m]-ideal_point[m] > epsilon):
        return individual.fitness.values[m] / (intercepts[m]-ideal_point[m])
    else:
        return individual.fitness.values[m] / epsilon

def normalize_objectives(individuals, intercepts, ideal_point):
    '''Normalizes individuals using the hyperplane defined by the intercepts as
    reference. Corresponds to Algorithm 2 of Deb & Jain (2014).'''
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        ind.fitness.normalized_values = list([normalize_objective(ind, m,
                                                                  intercepts, ideal_point)
                                                                  for m in range(num_objs)])
    return individuals

def perpendicular_distance(direction, point):
    k = np.dot(direction, point) / np.sum(np.power(direction, 2))
    d = np.sum(np.power(np.subtract(np.multiply(direction, [k] * len(direction)), point) , 2))
    return np.sqrt(d)

def associate(individuals, reference_points):
    '''Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).'''
    pareto_fronts = tools.sortLogNondominated(individuals, len(individuals))
    num_objs = len(individuals[0].fitness.values)

    for ind in individuals:
        rp_dists = [(rp, perpendicular_distance(ind.fitness.normalized_values, rp))
                    for rp in reference_points]
        best_rp, best_dist = sorted(rp_dists, key=lambda rpd:rpd[1])[0]
        ind.reference_point = best_rp
        ind.ref_point_distance = best_dist
        best_rp.associations_count +=1 # update de niche number
        best_rp.associations += [ind]
    
def Eud_distance(x,ideal_point):
    num_objs = len(x)
    dist = 0
    for i in range(num_objs):
        dist = dist + (x[i]-ideal_point[i])**2
    edi = np.sqrt(dist)
    return edi

def individual_angle(x,y):
    top = np.dot(x.fitness.normalized_values,y.fitness.normalized_values)
    bottom = np.linalg.norm(x.fitness.normalized_values)*np.linalg.norm(y.fitness.normalized_values)
    I_d = np.arccos(np.abs(top/bottom))
    return I_d

def individual_to_aixs_angle(x,y):
    top = np.dot(x.fitness.normalized_values,y)
    bottom = np.linalg.norm(x.fitness.normalized_values)*np.linalg.norm(y)
    I_d = np.arccos(np.abs(top/bottom))
    return I_d
    
def local_density(pop,pop_size):
    LD = np.zeros(pop_size)
    theta = np.pi/2
    for i in range(pop_size):
        for j in range(pop_size):
            I_d = individual_angle(pop[i],pop[j])
            if i !=j and I_d<theta:
                LD[i] += math.exp(-(I_d/theta)**2)
    return LD

def separation_distance(pop,pop_size):
    SD = np.ones(pop_size)*(np.pi/2)
    for i in range(pop_size):
        for j in range(pop_size):
            I_d = individual_angle(pop[i],pop[j])
            if I_d < SD[i]:
                SD[i] = I_d
    return SD
    
def Initialize_ADM(pop,pop_size,obj_num,theta_c):
    flag = np.zeros(pop_size)
    e = np.array(([1,0,0],[0,1,0],[0,0,1]))
    centroids = np.zeros((pop_size,obj_num))    
    for i in range(obj_num):
        temp = np.pi/2
        mark = 0
        for j in range(pop_size):
            I_d = individual_to_aixs_angle(pop[j],e[i])
            if I_d < temp:
                temp = I_d
                centroids[i] = pop[j].fitness.normalized_values
                mark = j
        flag[mark] = 1
    sort_index = np.argsort(-theta_c[1])
    
    i = obj_num
    j = 0
    while i < pop_size:
        if flag[j] == 0:
            centroids[i] = pop[sort_index[j]].fitness.normalized_values
            i = i + 1
        j = j + 1
    return centroids
    
def Self_guided_Adjustment(pop,pop_size,obj_num,centroids,ideal_point):
    C = []
    for i in range(pop_size):
        C.append([])
        
    cl = np.zeros(pop_size)
    for i in range(pop_size):
        temp = np.pi/2
        for j in range(pop_size):
            I_d = individual_to_aixs_angle(pop[i],centroids[j])
            if I_d<temp:
                temp = I_d
                cl[i] = j
        C[int(cl[i])].append(pop[int(cl[i])])
        
    t = 0
    flag = False
    centroids = centroids.tolist()

    while t<=2*obj_num and flag == False:
        for i in range(obj_num):
            C[i] = []
        l = 0
        for i in range(pop_size - obj_num):
            temp = 0
            edi = Eud_distance(pop[i+obj_num].fitness.normalized_values,ideal_point)
            if len(C[i+obj_num])!= 0 :
                for indi in C[i+obj_num]:
                    temp += indi.fitness.normalized_values/edi
                centroids[i+obj_num] = temp/len(C[i+obj_num])
            C[i+obj_num] = []
        
        for i in range(pop_size):
            temp = np.pi/2
            mark = 0 
            for j in range(pop_size):
                I_d = individual_to_aixs_angle(pop[i],centroids[j])
                if I_d<temp:
                    temp = I_d
                    mark = j
            if cl[i] != mark:
                cl[i] = mark
                l = l + 1   
            C[int(cl[i])].append(pop[i])
        if l == 0:
            flag = True  
        t = t + 1
    
    return centroids
        
def SRV(pop,pop_size,obj_num):
    ideal_point = find_ideal_point(pop)
    extremes = find_extreme_points(pop)
    intercepts = construct_hyperplane(pop, extremes)      
    normalize_objectives(pop, intercepts, ideal_point)
    
    LD = local_density(pop,pop_size)
    SD = separation_distance(pop,pop_size)
    theta_c = [LD,SD]
    
    centroids = Initialize_ADM(pop,pop_size,obj_num,theta_c)
    centroids = Self_guided_Adjustment(pop,pop_size,obj_num,centroids,ideal_point)
    
    srv = np.zeros((pop_size,obj_num))
    for i in range(pop_size):
        edi = Eud_distance(centroids[i],ideal_point)
        for j in range(obj_num):
            srv[i][j]= centroids[i][j]/edi
    return srv