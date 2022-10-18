import random
import numpy
import math
from solution import solution
import time
import copy
from sklearn.model_selection import KFold

def relu1(data):
    return numpy.maximum(data, 0.001)

def CaculateFitness(X,fun, s, c,train_x, train_y, test_x, test_y,data, labels):
    pop = X.shape[0]
    dim = X.shape[1]
    RRX = numpy.zeros([pop,1])
    RMSE = numpy.zeros([pop,1])
    MAE = numpy.zeros([pop,1])
    fitness = numpy.zeros([pop, 1])
    KF = KFold(n_splits = 10, shuffle = True, random_state = 520)
    for i in range(pop):
        RMSEL = []
        for k, (train, test) in enumerate(KF.split(data, labels)):
            train_x1 = data[train]
            test_x1 = data[test]
            train_y1 = labels[train]
            test_y1 = labels[test]
            _, test_MSE, _,_,_,_ =fun(train_x1,train_y1,test_x1,test_y1,s,c,X[i,0],X[i,1],X[i,2])
            RMSEL.append(test_MSE)
    # if test_RR2 > 0:
    #     RRL.append(test_RR2)
    # if train_RR2 > 0:
    #     train_RRL.append(train_RR2)
        fitness[i] = numpy.mean(RMSEL)
    return fitness,RRX,RMSE,MAE


def SSA(objf, lb, ub, dim, N, Max_iteration,s1,c,train_x,train_y,test_x,test_y,seed,data,labels):

    # Max_iteration=1000
    # lb=-100
    # ub=100
    # dim=30
    # N = 50  # Number of search agents
    # if not isinstance(lb, list):
    #     lb = [lb] * dim
    # if not isinstance(ub, list):
    #     ub = [ub] * dim
    numpy.random.seed(seed)
    Convergence_curve = numpy.zeros(Max_iteration)
    Convergence_curve2 = numpy.zeros(Max_iteration)
    IterPosition = numpy.zeros([Max_iteration,dim])
    first_Positions = numpy.zeros([N,dim])

    # Initialize the positions of salps
    SalpPositions = numpy.zeros((N, dim))
    for i in range(dim):
        # SalpPositions[:, i] = numpy.random.uniform(0, 1, N) * (ub[i] - lb[i]) + lb[i]
        SalpPositions[:, i] = numpy.random.random_integers(lb[i], ub[i], N)
    
    first_Positions = copy.copy(SalpPositions)
    print(SalpPositions)
    SalpFitness = numpy.full(N, float("inf"))

    FoodPosition = numpy.zeros(dim)
    FoodFitness = float("inf")
    # BestFitness = 0.001
    # BestPosition = [0,0,0]
    
    # Moth_fitness=numpy.fell(float("inf"))

    s = solution()

    # print('SSA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # for i in range(0, N):
    #     # evaluate moths
    #     # SalpFitness[i] = objf(SalpPositions[i, :])
    #     SalpFitness[i], _, _, _ = CaculateFitness(SalpPositions, objf, s1, c, train_x, train_y, test_x, test_y)
    SalpFitness, _, _, _ = CaculateFitness(SalpPositions, objf, s1, c, train_x, train_y, test_x, test_y,data,labels)

    sorted_salps_fitness = numpy.sort(SalpFitness)
    I = numpy.argsort(SalpFitness)  #从小到大

    Sorted_salps = numpy.copy(SalpPositions[I, :])

    FoodPosition = numpy.copy(Sorted_salps[0, :])
    FoodPosition = FoodPosition.reshape(-1,1)
    FoodFitness = sorted_salps_fitness[0]

    Iteration = 0

    # Main loop
    while Iteration < Max_iteration:
        
        # timerStart1 = time.time()
        # Number of flames Eq. (3.14) in the paper
        # Flame_no=round(N-Iteration*((N-1)/Max_iteration));

        c1 = 2 * math.exp(-((4 * Iteration / Max_iteration) ** 2))
        # Eq. (3.2) in the paper

        SalpPositions = numpy.transpose(SalpPositions)
        for i in range(0, N):

            if i < N / 2:
                for j in range(0, dim):
                    c2 = random.random()
                    c3 = random.random()
                    # Eq. (3.1) in the paper
                    if c3 < 0.5:
                        SalpPositions[j, i] = FoodPosition[j] + c1 * ((ub[j] - lb[j]) * c2 + lb[j])
                    else:
                        SalpPositions[j, i] = FoodPosition[j] - c1 * ((ub[j] - lb[j]) * c2 + lb[j])

                    ####################

            elif i >= N / 2 and i < N + 1:
                point1 = SalpPositions[:, i - 1]
                point2 = SalpPositions[:, i]

                SalpPositions[:, i] = (point2 + point1) / 2
                # Eq. (3.4) in the paper

        SalpPositions = numpy.transpose(SalpPositions)

        for i in range(0, N):

            # Check if salps go out of the search spaceand bring it back
            for j in range(dim):
                SalpPositions[i, j] = numpy.clip(SalpPositions[i, j], lb[j], ub[j])

        SalpFitness, _, _, _ = CaculateFitness(SalpPositions, objf, s1, c, train_x, train_y, test_x, test_y,data,labels)

        for i in range(0, N):
            # for j in range(dim):
            if SalpFitness[i] < FoodFitness:
                FoodPosition = numpy.copy(SalpPositions[i, :])
                # FoodPosition = FoodPosition.reshape(-1, 1)
                FoodFitness = SalpFitness[i]
                    

        # Display best fitness along the iteration
        if Iteration % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(Iteration+1)
                    + " the best fitness is "
                    + str(FoodFitness)
                ]
            )
            FoodPosition1= [int(i) for i in FoodPosition]
            print(FoodPosition1)
            if Iteration == 0:
                first_Food = copy.copy(FoodPosition1)
        
        
        IterPosition[Iteration,:] = copy.copy(FoodPosition1)

        ## 计算平均适应度
        Fitnessmean = numpy.median(SalpFitness)
        Convergence_curve2[Iteration] = Fitnessmean
        
        
        # if 1/FoodFitness > BestFitness:
        #     BestFitness = 1/FoodFitness
        #     BestPosition = FoodPosition
        Convergence_curve[Iteration] = 1/FoodFitness
        s.best = 1/FoodFitness
        s.bestIndividual = FoodPosition

        Iteration = Iteration + 1
        
        timerEnd1 = time.time()
        # execution_Time = timerEnd1 - timerStart
        Cumulative_time = timerEnd1 - timerStart
        print('Cumulative_time:',Cumulative_time)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "SSA"
    # s.objfname = objf.__name__

    return s,Convergence_curve2,IterPosition,first_Positions