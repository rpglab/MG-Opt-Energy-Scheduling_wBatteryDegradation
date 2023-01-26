# -*- coding: utf-8 -*-
"""
Created on Wen Aug 18 15:42:20 2021

@author: awesomezhao
"""


from pyomo.environ import *
import pandas as pd
#import gurobipy
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from operator import itemgetter
from itertools import groupby

start = time.time()
print("hello")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

Loopsize = 60
ReducePercentage = 0.05
TotalDegradation = []
DegradationCost = []
OperationalCost = []
RemainingCapacity = []
dflist = []
for K in range(1,Loopsize+1):
    # instance the pyomo abstract model
    model = AbstractModel()
    
    ### set
    model.PERIOD = Set()
    model.GEN = Set()
    
    ### Generater Parameters
    model.gen_min = Param(model.GEN)
    model.gen_max = Param(model.GEN)
    model.gen_RRlimit = Param(model.GEN)
    model.gen_Kom = Param(model.GEN)
    model.gen_NlCost = Param(model.GEN)
    model.gen_SuCost = Param(model.GEN)
    
    ### Constant Parameters
    model.Ppv = Param(default=20)
    model.Pwt = Param(default=2)
    model.Esslimit_Max = Param(default=300)
    model.Esslimit_Min = Param(default=0)
    model.Gridlimit = Param(default=1500)
    model.DeltaE = Param(default=30)
    model.DeltaPgrid = Param(default=50)
    model.DeltaPess = Param(default=0)
    model.Pess_Max = Param(default=150)#*pow((1-ReducePercentage),(K-1)))
    model.Pess_Min = Param(default=0)
    model.ESSinitial = Param(default=200)
    #model.ESSinitial = Var()
    model.R = Param(default = 0.03)
    # model.degra =Param(default = 0.002)
    # model.Cost_SOC =Param(default=20)
    model.ESSE = Param(default=0.9)
    model.ESSFeedBackLimit = Param(default = 1000*pow((1-ReducePercentage),(K-1)))
    model.ESSFeedBackLimit1 = Param(default = 750*pow((1-ReducePercentage),(K-1)))
    
    ### cost parameters
    model.PgridCost = Param(model.PERIOD)
    model.SelltoGrid = Param(default = 0.8)
    
    ### period parameters
    model.Time_TotalPd = Param(model.PERIOD)
    model.PV_Ava = Param(model.PERIOD)
    model.WT_Ava = Param(model.PERIOD)
    model.BigM = Param(default = 1000)
    
    ### Variables
    model.u = Var(model.GEN,model.PERIOD, within =Binary)
    model.v = Var(model.GEN,model.PERIOD, within =Binary)
    model.USCHARGE = Var(model.PERIOD, within =Binary)
    model.USDISCHARGE = Var(model.PERIOD, within =Binary)
    model.UBUYFGRID = Var(model.PERIOD, within =Binary)
    model.USELLTGRID = Var(model.PERIOD, within =Binary)
    model.Vg = Var(model.PERIOD, within =Binary)
    model.Pg = Var(model.GEN, model.PERIOD)
    model.Grid = Var(model.PERIOD)
    model.BUYFGRID = Var(model.PERIOD)
    model.SELLTGRID = Var(model.PERIOD)
    model.PCHARGE = Var(model.PERIOD)
    model.PDISCHARGE = Var(model.PERIOD)
    model.ESS = Var(model.PERIOD)
    model.PPV = Var(model.PERIOD)
    model.PWT = Var(model.PERIOD)
    model.SOC = Var(model.PERIOD)
    model.Temp = Param(model.PERIOD)
    
    
    ###objective function
    def objfunction(model):
        totalcost = sum(model.gen_Kom[i]*(model.Pg[i,j] - model.gen_min[i])+ model.gen_NlCost[i]*model.u[i,j]+ model.gen_SuCost[i]*model.v[i,j] for i in model.GEN for j in model.PERIOD) + sum(model.BUYFGRID[j]*model.PgridCost[j]-model.SELLTGRID[j]*model.PgridCost[j]*model.SelltoGrid for j in model.PERIOD) #+ sum(model.PCHARGE[j]*model.ESSE*model.degra + model.PDISCHARGE[j]*model.ESSE*model.degra + (model.SlackLow[j]+model.SlackHigh[j])*model.Cost_SOC for j in model.PERIOD)
        return totalcost
    model.Cost = Objective(rule = objfunction, sense = minimize)
    
    ### Constrains
    #power balance
    
    def powerbalance(model,j):
        power = sum(model.Pg[i,j] for i in model.GEN) + model.BUYFGRID[j]+ model.PDISCHARGE[j] + model.Ppv*model.PV_Ava[j] + model.Pwt*model.WT_Ava[j] 
        power = power - (model.Time_TotalPd[j] + model.SELLTGRID[j] + model.PCHARGE[j])
        return  power == 0
    model.Cons_powerblance = Constraint(model.PERIOD, rule = powerbalance)
    
    def ESSbalance(model,j):
        if j >=2:    
            expr = model.ESS[j] -model.ESS[j-1] + model.PDISCHARGE[j]/model.ESSE-model.PCHARGE[j]*model.ESSE
            return expr ==0
        else:
            return Constraint.Skip
    model.Cons_ESSBalance = Constraint(model.PERIOD, rule = ESSbalance)
    
    def ESSinitial(model, j):
        expr = model.ESS[1] -model.ESSinitial + model.PDISCHARGE[1]-model.PCHARGE[1]
        return expr == 0
    model.Cons_ESSinitial = Constraint(model.PERIOD, rule = ESSinitial)
    
    # def ESSinitialValue(model):
    #     expr = model.ESSinitial
    #     return expr <= 1
    # model.Cons_ESSinitialValue = Constraint(rule = ESSinitialValue)

    # def ESSinitialValue1(model):
    #     expr = model.ESSinitial
    #     return expr >= 0
    # model.Cons_ESSinitialValue1 = Constraint(rule = ESSinitialValue1)
    
    
    def SOC(model,j):
        expr = model.ESS[j]/model.Esslimit_Max - model.SOC[j]
        return expr == 0
    model.Cons_SOC = Constraint(model.PERIOD, rule = SOC)
    
    def ESSReduceUsage(model,j):
        expr = sum(model.PCHARGE[j] for j in model.PERIOD)+ sum(model.PDISCHARGE[j] for j in model.PERIOD)
        return expr <= model.ESSFeedBackLimit
    model.Cons_ESSReduceUsage = Constraint(model.PERIOD, rule = ESSReduceUsage)
   
    # def ESSReduceUsage2(model,j):
    #     expr = model.PCHARGE[3] + model.PCHARGE[4] + model.PCHARGE[24] + model.PDISCHARGE[1] + model.PDISCHARGE[21]
    #     return expr <= model.ESSFeedBackLimit1
    # model.Cons_ESSReduceUsage2 = Constraint(model.PERIOD, rule = ESSReduceUsage2)
    
    def ESSkeepon(model,j):
        expr =  model.USCHARGE[j] + model.USDISCHARGE[j]
        return expr <= 1
    model.Cons_ESSKeepon = Constraint(model.PERIOD, rule = ESSkeepon)
    
    def Essend(model, j):
        expr = model.ESS[24]
        return expr == model.ESSinitial
    model.Cons_Essend = Constraint(model.PERIOD, rule = Essend)
    
    def backup(model, j):
        return model.R*model.Time_TotalPd[j] <= model.Gridlimit -model.BUYFGRID[j] + model.SELLTGRID[j] + sum(model.u[i,j]*model.gen_max[i]-model.Pg[i,j] for i in model.GEN for j in model.PERIOD)
    model.Cons_backup =Constraint(model.PERIOD,rule = backup)
    
    def genlimit1(model,i, j):
        return model.u[i,j]*model.gen_min[i] <= model.Pg[i,j]
    model.Cons_genMax1 = Constraint(model.GEN,model.PERIOD,rule = genlimit1)
    
    def genlimit2(model,i, j):
        return  model.Pg[i,j]<= model.u[i,j]*model.gen_max[i]
    model.Cons_genMax2 = Constraint(model.GEN,model.PERIOD,rule = genlimit2)
    
    def genramp1(model,i,j):
        if j >=2:
            expr = model.Pg[i,j] - model.Pg[i,j-1] - model.gen_RRlimit[i]
            return  expr <=0
        else:
            return Constraint.Skip
    model.Cons_genramp1 = Constraint(model.GEN,model.PERIOD,rule = genramp1)
    
    def genramp2(model,i,j):
        if j >=2:
            expr = model.Pg[i,j-1] - model.Pg[i,j] - model.gen_RRlimit[i]
            return   expr <=0
        else:
            return Constraint.Skip
    model.Cons_genramp2 = Constraint(model.GEN,model.PERIOD,rule = genramp2)
    
    
    def Esslimit1(model,j):
        return  model.ESS[j] <= model.Esslimit_Max
    model.Cons_ESSlimit1 = Constraint(model.PERIOD,rule = Esslimit1)
    
    def Esslimit2(model,j):
        return model.Esslimit_Min <= model.ESS[j]
    model.Cons_ESSlimit2 = Constraint(model.PERIOD,rule = Esslimit2)
    
    def PCHARGElimit1(model,j):
        return model.USCHARGE[j]*(model.Pess_Min) <= model.PCHARGE[j] 
    model.Cons_PCHARGElimit1 = Constraint(model.PERIOD,rule = PCHARGElimit1)
    
    def PCHARGElimit2(model,j):
        return  model.PCHARGE[j] <= model.USCHARGE[j]*(model.Pess_Max-model.DeltaPess)
    model.Cons_PCHARGElimit2 = Constraint(model.PERIOD,rule = PCHARGElimit2)
    
    def PDISCHARGElimit1(model,j):
        return model.USDISCHARGE[j]*(model.Pess_Min) <= model.PDISCHARGE[j] 
    model.Cons_PDISCHARGElimit1 = Constraint(model.PERIOD,rule = PDISCHARGElimit1)
    
    def PDISCHARGElimit2(model,j):
        return model.PDISCHARGE[j] <= model.USDISCHARGE[j]*(model.Pess_Max-model.DeltaPess)
    model.Cons_PDISCHARGElimit2 = Constraint(model.PERIOD,rule = PDISCHARGElimit2)
    
    def BGridlimit1(model,j):
        return 0 <= model.BUYFGRID[j]
    model.Cons_BGridlimit1 = Constraint(model.PERIOD,rule = BGridlimit1)
    
    def BGridlimit2(model,j):
        return model.BUYFGRID[j] <= model.UBUYFGRID[j]*(model.Gridlimit - model.DeltaPgrid)
    model.Cons_BGridlimit2 = Constraint(model.PERIOD,rule = BGridlimit2)
    
    def SGridlimit1(model,j):
        return 0 <= model.SELLTGRID[j]
    model.Cons_SGridlimit1 = Constraint(model.PERIOD,rule = SGridlimit1)
    
    def SGridlimit2(model,j):
        return model.SELLTGRID[j] <= model.USELLTGRID[j]*(model.Gridlimit - model.DeltaPgrid)
    model.Cons_SGridlimit2 = Constraint(model.PERIOD,rule = SGridlimit2)
    
    def GridStatus(model,j):
        return model.UBUYFGRID[j] + model.USELLTGRID[j] <= 1
    model.Cons_GridStatus = Constraint(model.PERIOD,rule = GridStatus)
    
    def GridExchange(model,j):
        return model.Grid[j] == model.BUYFGRID[j]-model.SELLTGRID[j]
    model.Cons_GridExchange = Constraint(model.PERIOD,rule =GridExchange)
     
    
    def genUV1(model,i,j):
        if j >=2:
            expr = model.u[i,j] - model.u[i,j-1] - model.v[i,j]
            return expr <= 0
        else:
            return Constraint.Skip
    model.Cons_genUV1 = Constraint(model.GEN,model.PERIOD,rule = genUV1)
    
    def genUV2(model,i,j):
        if j >=2:
            expr = model.u[i,j-1] - model.u[i,j] - model.v[i,j]
            return expr <= 0
        else:
            return Constraint.Skip
    model.Cons_genUV2 = Constraint(model.GEN,model.PERIOD,rule = genUV2)
    
    def geninitial(model,i):
        expr = model.u[i,1] - model.v[i,1]
        return expr <= 0
    model.Cons_geninitial = Constraint(model.GEN, rule = geninitial)
    
    
    # instance according on the dat file
    SCUC_instance = model.create_instance('case16_nolayers.dat')
    
    ### set the solver
    SCUCsolver = SolverFactory('gurobi')
    SCUCsolver.options.mipgap = 0.0
    results = SCUCsolver.solve(SCUC_instance)
    Data =[]
    genunit =[]
    
    print("\nresults.Solution.Status: " + str(results.Solution.Status))
    print("\nresults.solver.status: " + str(results.solver.status))
    print("\nresults.solver.termination_condition: " + str(results.solver.termination_condition))
    print("\nresults.solver.termination_message: " + str(results.solver.termination_message))
    print('\nminimize cost: ' + str(SCUC_instance.Cost()))
    
    for j in SCUC_instance.PERIOD:
        X = [str(SCUC_instance.BUYFGRID[j]()), str(SCUC_instance.SELLTGRID[j]()), str(SCUC_instance.PDISCHARGE[j]()-SCUC_instance.PCHARGE[j]()),str(SCUC_instance.ESS[j]()),str(SCUC_instance.SOC[j]()),int(SCUC_instance.USDISCHARGE[j]()),int(SCUC_instance.USCHARGE[j]())]
        Data.append(X)
        #Y = [str(SCUC_instance.u[1,j]()),str(SCUC_instance.v[1,j]()),str(SCUC_instance.Pg[1,j]()),str(SCUC_instance.u[2,j]()),str(SCUC_instance.v[2,j]()),str(SCUC_instance.Pg[2,j]()),str(SCUC_instance.u[3,j]()),str(SCUC_instance.v[3,j]()),str(SCUC_instance.Pg[3,j]())]
        Y = [str(SCUC_instance.Pg[1,j]()),str(SCUC_instance.Pg[2,j]()),str(SCUC_instance.Pg[3,j]())]
        genunit.append(Y)
    Data = pd.DataFrame(Data, columns=['BUYFGRID', 'SELLTGRID', 'BESS','ESS','SOC','UD','UC'])
    genunit =pd.DataFrame(genunit,columns=['pg1','pg2','pg3'])
    #Z = sum(float(Data['CHARGE']))
    print (genunit)
    print (Data)
    
    dflist = [dflist,Data]
    
    #degradation prediction function
    
    def BatteryDegradation(Capacity, Temp, DISC, SOCL, SOCH, Type):
        
        input_x = np.zeros((1,6))
        input_x[0,0] = Capacity
        input_x[0,1] = Temp
        input_x[0,2] = DISC
        input_x[0,3] = SOCL
        input_x[0,4] = SOCH
        input_x[0,5] = Type
        
        x_tensor = torch.Tensor(input_x).unsqueeze(0).cuda()  # unsqueeze gives a 1, batch_size dimension
        
        hidden = None
        #test_out, test_h = load_rnn(x_tensor, hidden)          # use trained_rnn not load_rnn
        test_out = load_rnn(x_tensor)
        Degradation = test_out.cpu().data.numpy().flatten()
        return Degradation
    
    # load_model]
    
    load_rnn = torch.load('trained_nn_1d.pt')
    
    Degradation =[]
    SingleD = []
    TotalCap = 300 # maximum capacity of BESS
    Capacity = 0.9 # we assume the battery capacity is 90%
    Type = 1
    K = Data['UD'].to_numpy().nonzero()
    data = np.asarray(K).flatten().tolist()
    ranges = []
    SUM = 0
    
    #Cycle Based Battery Usage Processing Method for discharging cycle
    for k, g in groupby(enumerate(data), lambda i: i[0] - i[1]):
        SUM = 0
        input_x = np.zeros((1,6))
        group = list(map(itemgetter(1), g))
        #print(group)
        SIZE = np.size(group)
        for countnumbner, INDEX in enumerate(group):
            SUM = SUM + abs(float(Data.iloc[int(INDEX)]['BESS']))
        Average = SUM/SIZE
        #print(Average)
        if group[0] == 0:
            input_x[0,0] = Capacity
            input_x[0,1] = 25/32
            input_x[0,2] = Average/TotalCap/2
            input_x[0,3] = 0.667
            input_x[0,4] = SUM/TotalCap
            input_x[0,5] = Type   
        else:
            input_x[0,0] = Capacity
            input_x[0,1] = 25/32
            input_x[0,2] = Average/TotalCap/2
            input_x[0,3] = float(Data.iloc[int(group[0])-1]['SOC'])
            input_x[0,4] = SUM/TotalCap
            input_x[0,5] = Type 
        x_tensor = torch.Tensor(input_x).unsqueeze(0).cuda()
        hidden = None
        #test_out, test_h = load_rnn(x_tensor, hidden)
        test_out = load_rnn(x_tensor)
        SingleD = test_out.cpu().data.numpy().flatten()/2
        print('\nDegradation retreived from NN for' + str(group) + 'is :'  + str(SingleD))
        Degradation = np.append(Degradation, SingleD)    
    
    K = Data['UC'].to_numpy().nonzero()
    data = np.asarray(K).flatten().tolist()
    ranges = []
    SUM = 0
    Capacity = 0.9 # we assume the battery capacity is 90%
    Type = 1
    #Cycle Based Battery Usage Processing Method for discharging cycle
    for k, g in groupby(enumerate(data), lambda i: i[0] - i[1]):
        SUM = 0
        input_x = np.zeros((1,6))
        group = list(map(itemgetter(1), g))
        #print(group)
        SIZE = np.size(group)
        for countnumbner, INDEX in enumerate(group):
            SUM = SUM + abs(float(Data.iloc[int(INDEX)]['BESS']))
        Average = SUM/SIZE
        #print(Average)
        if group[0] == 0:
            input_x[0,0] = Capacity
            input_x[0,1] = 25/32
            input_x[0,2] = Average/TotalCap/2
            input_x[0,3] = 0.667
            input_x[0,4] = SUM/TotalCap
            input_x[0,5] = Type   
        else:
            input_x[0,0] = Capacity
            input_x[0,1] = 25/32
            input_x[0,2] = Average/TotalCap/2
            input_x[0,3] = float(Data.iloc[int(group[0])-1]['SOC'])
            input_x[0,4] = SUM/TotalCap
            input_x[0,5] = Type 
        x_tensor = torch.Tensor(input_x).unsqueeze(0).cuda()
        hidden = None
        #test_out, test_h = load_rnn(x_tensor, hidden)
        test_out = load_rnn(x_tensor)
        SingleD = test_out.cpu().data.numpy().flatten()/2
        print('\nDegradation retreived from NN for' + str(group) + 'is :'  + str(SingleD))
        Degradation = np.append(Degradation, SingleD) 
        #print(SingleD)
    TotalDegradation = np.append(TotalDegradation, sum(Degradation)/100)
    RemainingCapacity = np.append(RemainingCapacity,90-sum(Degradation)/100)
    DegradationCost = np.append(DegradationCost, sum(Degradation)*TotalCap*400/10000*2)
    OperationalCost = np.append(OperationalCost, SCUC_instance.Cost())
    #print(TotalDegradation(K))
    #print(DegradationCost(K))
    
    #determine when to stop the iteration
    
    # if K>=10:
    #     if DegradationCost[K-9]+OperationalCost[K-9]-DegradationCost[K-10]-OperationalCost[K-10]<=0:
    #         if DegradationCost[K-8]+OperationalCost[K-8]-DegradationCost[K-9]-OperationalCost[K-9]<=0:
    #             if DegradationCost[K-7]+OperationalCost[K-7]-DegradationCost[K-8]-OperationalCost[K-8]<=0:
    #                 if DegradationCost[K-6]+OperationalCost[K-6]-DegradationCost[K-7]-OperationalCost[K-7]<=0:
    #                     if DegradationCost[K-5]+OperationalCost[K-5]-DegradationCost[K-6]-OperationalCost[K-6]>=0:
    #                         if DegradationCost[K-4]+OperationalCost[K-4]-DegradationCost[K-5]-OperationalCost[K-5]>=0:
    #                             if DegradationCost[K-3]+OperationalCost[K-3]-DegradationCost[K-4]-OperationalCost[K-4]>=0:
    #                                 if DegradationCost[K-2]+OperationalCost[K-2]-DegradationCost[K-3]-OperationalCost[K-3]>=0:
    #                                     if DegradationCost[K-1]+OperationalCost[K-1]-DegradationCost[K-2]-OperationalCost[K-2]>=0:
    #                                         break

    # else:
    #     pass


####figure plot setup

TotalCost = DegradationCost + OperationalCost

Loop = np.linspace(1,len(TotalCost),len(TotalCost))

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(Loop, DegradationCost)
axs[0, 0].set_title("Degradation Cost Curve ($)")
axs[1, 0].plot(Loop, TotalDegradation)
axs[1, 0].set_title("Battery Total Degrdation Per day %")
axs[1, 0].sharex(axs[0, 0])
axs[0, 1].plot(Loop, OperationalCost)
axs[0, 1].set_title("Operation Cost Curve ($)")
axs[1, 1].plot(Loop, TotalCost)
axs[1, 1].set_title("Total Cost Curve ($)")
#fig.suptitle('SCUC Result with {} iterations of MCA-BCL '.format(Loopsize))
fig.tight_layout()
#fig.savefig('Updated_BCL.svg')

index = np.where(TotalCost == TotalCost.min())
min_index = index[0]
DecreaseTotalCost = (TotalCost[0]-TotalCost[min_index])/TotalCost[0]
DecreaseTotalDegradation = (TotalDegradation[0]-TotalDegradation[min_index])/TotalDegradation[0]
DecreaseBatteryCost = (DegradationCost[0]-DegradationCost[min_index])/DegradationCost[0]
IncreaseOperationCost = (OperationalCost[min_index]-OperationalCost[0])/OperationalCost[min_index]

print('Minimum Total Cost is $ {}'.format(TotalCost[min_index]))   
print('Maximum Degradation Cost is $ {}'.format(max(DegradationCost)))
print('DecreaseTotalCost is $ {}'.format(DecreaseTotalCost))
print('DecreaseTotalDegradation is {} %%'.format(DecreaseTotalDegradation))
print('Decreased Degradation Cost is $ {} '.format(DecreaseBatteryCost))
print('IncreaseOperationCost is $ {} '.format(IncreaseOperationCost))
print('Related Interation number is $ {} '.format(min_index))
end = time.time()
print('Time is {}'.format(end - start))

outputresult = [TotalCost[min_index], DegradationCost[min_index], DecreaseTotalCost, DecreaseTotalDegradation, IncreaseOperationCost, end - start]












