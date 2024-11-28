"""
快速非支配排序 
"""
import random 
import numpy as np
import math
from rdkit import Chem, DataStructs
def nonDominationSort(pops, fits):
    """快速非支配排序算法，最大化目标
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
    Return:
        ranks: 每个个体所对应的等级，一维数组

    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # 目标函数的个数
    ranks = np.zeros(nPop, dtype=np.int32)
    nPs = np.zeros(nPop)  # 每个个体p被支配解的个数
    sPs = []  # 每个个体支配的解的集合，把索引放进去
    for i in range(nPop):
        iSet = []  # 解i的支配解集
        for j in range(nPop):
            if i == j:
                continue
            isDom1 = fits[i] >= fits[j] #原<=改为越大越好
            isDom2 = fits[i] > fits[j] #原<
            # 是否支配该解-> i支配j
            if sum(isDom1) == nF and sum(isDom2) >= 1:
                iSet.append(j)
            # 是否被支配-> i被j支配
            if sum(~isDom2) == nF and sum(~isDom1) >= 1:
                nPs[i] += 1
        sPs.append(iSet)  # 添加i支配的解的索引
    r = 0  # 当前等级为 0， 等级越低越好
    indices = np.arange(nPop)
    while sum(nPs==0) != 0:
        rIdices = indices[nPs==0]  # 当前被支配数为0的索引
        ranks[rIdices] = r
        for rIdx in rIdices:
            iSet = sPs[rIdx]
            nPs[iSet] -= 1
        nPs[rIdices] = -1  # 当前等级的被支配数设置为负数
        r += 1
    return ranks

def level_p(fits, L):
    """快速非支配排序算法，最大化目标
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
        L: 划分区域的间隔数
    Return:
        ranks: 每个个体所对应的等级，一维数组

    """
    nPop = fits.shape[0]
    nF = fits.shape[1]  # 目标函数的个数

    level = np.zeros(nPop, dtype=np.int32)#存储每个分子的等级

    for i in range(nPop):#每个分子
        ll=[]#等级越小越好
        for f in range(nF):#每个目标所属区间
            deltaf = (max(fits[:,f]) - min(fits[:,f]))/L#仅一个分子deltaf=0
            if deltaf == 0:
                ll.append(L + 1)
            else:
                ll.append(int((max(fits[:, f]) - fits[i][f]) / deltaf) + 1)  # 分子i的第f个目标所属的等级

        level[i] = max(ll)#所有目标中所属等级最大的
    return level
# 拥挤度排序算法
def crowdingDistanceSort(pops, fits, ranks):
    """拥挤度排序算法
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组
        ranks：每个个体对应的等级，一维数组
    Return：
        dis: 每个个体的拥挤度，一维数组
    """
    nPop = pops.shape[0]
    nF = fits.shape[1]  # 目标个数
    dis = np.zeros(nPop)
    nR = ranks.max()  # 最大等级
    indices = np.arange(nPop)
    for r in range(nR+1):
        rIdices = indices[ranks==r]  # 当前等级种群的索引
        rPops = pops[ranks==r]  # 当前等级的种群
        rFits = fits[ranks==r]  # 当前等级种群的适应度
        rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引
        rSortFits = np.sort(rFits,axis=0)
        fMax = np.max(rFits,axis=0)
        fMin = np.min(rFits,axis=0)
        n = len(rIdices)
        for i in range(nF):
            orIdices = rIdices[rSortIdices[:,i]]  # 当前操作元素的原始位置
            j = 1
            while n > 2 and j < n-1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / \
                                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf
                j += 1
            dis[orIdices[0]] = np.inf
            dis[orIdices[n-1]] = np.inf
    return dis
# 拥挤度排序算法 
def crowding_level(pops, fits, ranks):
    """拥挤度排序算法
    Params:
        pops: 种群，nPop * nChr 数组
        fits: 适应度， nPop * nF 数组 
        ranks：每个个体对应的等级，一维数组 
    Return：
        dis: 每个个体的拥挤度，一维数组 
    """
    nPop = pops.shape[0] 
    nF = fits.shape[1]  # 目标个数 
    dis = np.zeros(nPop) 
    #nR = ranks.max()  # 最大等级
    uni_level = sorted(list(set(ranks)))#从小到大
    indices = np.arange(nPop) 
    for r in uni_level:
        rIdices = indices[ranks==r]  # 当前等级种群的索引
        rFits = fits[ranks==r]  # 当前等级种群的适应度 
        rSortIdices = np.argsort(rFits, axis=0)  # 对纵向排序的索引 
        rSortFits = np.sort(rFits,axis=0) 
        fMax = np.max(rFits,axis=0) 
        fMin = np.min(rFits,axis=0) 
        n = len(rIdices)
        for i in range(nF): 
            orIdices = rIdices[rSortIdices[:,i]]  # 当前操作元素的原始位置 
            j = 1  
            while n > 2 and j < n-1:
                if fMax[i] != fMin[i]:
                    dis[orIdices[j]] += (rSortFits[j+1,i] - rSortFits[j-1,i]) / \
                                        (fMax[i] - fMin[i])
                else:
                    dis[orIdices[j]] = np.inf 
                j += 1 
            dis[orIdices[0]] = np.inf 
            dis[orIdices[n-1]] = np.inf   
    return dis



'''
if __name__ == "__main__":
    y1 = np.arange(1,5).reshape(4,1)
    y2 = 5 - y1 
    fit1 = np.concatenate((y1,y2),axis=1) 
    y3 = 6 - y1 
    fit2 = np.concatenate((y1,y3),axis=1)
    y4 = 7 - y1 
    fit3 = np.concatenate((y1,y4),axis=1) 
    fit3 = fit3[:2] 
    fits = np.concatenate((fit1,fit2,fit3), axis=0) 
    pops = np.arange(fits.shape[0]).reshape(fits.shape[0],1) 

    
    random.seed(123)
    # 打乱数组
    indices = np.arange(fits.shape[0])
    random.shuffle(indices)
    fits = fits[indices]
    pops = pops[indices]
    print(indices) 

    # 首先测试非支配排序算法 
    ranks = nonDominationSort(pops, fits) 
    print('ranks:', ranks) 
    
    # 测试拥挤度排序算法 
    dis = crowdingDistanceSort(pops,fits,ranks) 
    print("dis:", dis) 
'''


def CDP_simple(pop, fits, CV):
    # 最大化目标函数
    n_pop = len(pop)
    rank = np.zeros(n_pop, dtype=int)
    # 储存i支配的个体列表
    S = [[] for i in range(n_pop)]
    # i被其他个体支配的个数
    N = np.zeros(n_pop, dtype=int)
    # 储存每层前沿面上的点
    #F = [[] for i in range(n_pop)]

    for i in range(n_pop):
        for j in range(n_pop):
            if i != j:
                # 不是两个可行解，判断支配关系
                if CV[i] < CV[j]:
                    S[i].append(j)
                elif CV[j] < CV[i]:
                    N[i] += 1
                else:
                # 两个可行解，判断支配关系
                    if all(fits[i] >= fits[j]) and any(fits[i] > fits[j]):
                        S[i].append(j)
                    elif all(fits[j] >= fits[i]) and any(fits[j] > fits[i]):
                        N[i] += 1
    '''
        if N[i] == 0:
            # 前沿，没有被其他解支配
            rank[i] = 1
            F[0].append(i)
    
    i = 0
    # 循环每个前沿面直至无解
    while len(F[i]) > 0:
        Q = []
        for p in F[i]:
            #
            for q in S[p]:
                N[q] -= 1
                if N[q] == 0:
                    rank[q] = i + 2
                    Q.append(q)
        i += 1
        F.append(Q)
    rank[rank == 0] = max(rank) + 1
    #将等级由0开始
    rank = rank-1
    '''
    r = 0  # 当前等级为 0， 等级越低越好
    indices = np.arange(n_pop)
    while sum(N == 0) != 0:
        rIdices = indices[N == 0]  # 当前被支配数为0的索引
        rank[rIdices] = r
        for rIdx in rIdices:
            iSet = S[rIdx]
            N[iSet] -= 1
        N[rIdices] = -1  # 当前等级的被支配数设置为负数
        r += 1
    return rank

def score2_CDP(MergePops_uni, MergeFits_uni, MergeCV_uni):
    '''

    Args:
        MergePops_uni: 种群
        MergeFits_uni: 适应度

    Returns: 根据适应度和拥挤度的种群个体得分

    '''
    MergeRanks = CDP_simple(MergePops_uni, MergeFits_uni, MergeCV_uni)
    MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni, MergeRanks)
    indices = np.arange(MergePops_uni.shape[0])
    Rank2 = np.zeros(MergePops_uni.shape[0]).astype(int)  # 储存pops排序的索引，排名第一是第几行
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while r <= max(MergeRanks):
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx]
        Rank2[i:i + len(rIndices)] = surIndices
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    S2 = np.zeros(MergePops_uni.shape[0])
    S2[Rank2] = np.arange(len(Rank2))#S2中分子ID对应位置分别赋予1,2,3.。。
    return S2

'''
MergePops_uni = np.random.rand(5, 3)
MergeFits_uni = np.random.rand(5, 2)
CV= np.array([1,0,0,2,1])

MergeRanks = CDP_simple(MergePops_uni, MergeFits_uni, CV)
MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni, MergeRanks)
indices = np.arange(MergePops_uni.shape[0])
Rank2 = np.zeros(MergePops_uni.shape[0]).astype(int)  # 储存pops排序的索引，排名第一是第几行
r = 0
i = 0
rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
while r <= max(MergeRanks):
    rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
    rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
    surIndices = rIndices[rSortedIdx]
    Rank2[i:i + len(rIndices)] = surIndices
    r += 1  # 当前等级+1
    i += len(rIndices)
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
S2 = np.zeros(MergePops_uni.shape[0])
S2[Rank2] = np.arange(len(Rank2))
print(S2)
'''
            
        


