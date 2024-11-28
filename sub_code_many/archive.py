from property import *
from nonDominationSort import *
import torch
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from rdkit.Chem import Descriptors

def archive_nsga2(pops, fits, nPop):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nChr = pops.shape[1]
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    #首先去除重复
    MergeRanks = nonDominationSort(pops, fits)
    MergeDistances = crowdingDistanceSort(pops, fits, MergeRanks)

    indices = np.arange(pops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) <= nPop:
        newPops[i:i + len(rIndices)] = pops[rIndices]
        newFits[i:i + len(rIndices)] = fits[rIndices]
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        #若加到最后一个等级仍不足种群数，随机采分子
        #不够的分子采样补全
        if r == max(MergeRanks)+1:
            if i < nPop:
                idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
                idx = compare(idx1, idx2, MergeRanks, MergeDistances)
                newPops[i] = pops[idx]
                newFits[i] = fits[idx]
                i += 1

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = pops[surIndices]
        newFits[i:] = fits[surIndices]
    return (newPops, newFits)

def compare(idx1, idx2, ranks, distances):
    # return: 更优的 idx
    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if distances[idx1] <= distances[idx2]:
            idx = idx2
        else:
            idx = idx1
    return idx

def archive_stage1(MergePops, MergeFits, MergeCV, Mergesmiles, nPop,
                  Archive_pops,Archive_fits,Archive_CV, Arc_smiles, iter):
    """
    Args:
        pops: 种群嵌入
        fits: 种群适应度
        chrpops: 子代嵌入
        chrfits: 子代适应度
        CV_pops: 种群约束违反度
        CV_offspring: 子代约束违反度
        nPop: 种群大小
        Archive_pops: 档案库分子嵌入
        Archive_fits: 档案库分子适应度
        Archive_CV: 档案库分子约束违反度
        iter: 当前迭代

    Returns:

    """
    #合并种群的属性和违反度
    if iter>1:
        # 合并种群和先前archive的属性和违反度
        MergePops = np.concatenate((Archive_pops, MergePops), axis=0)
        MergeFits = np.concatenate((Archive_fits, MergeFits), axis=0)
        MergeCV = np.concatenate((Archive_CV, MergeCV), axis=0)
        Mergesmiles = np.concatenate((Arc_smiles, Mergesmiles), axis=0)
    #去除重复属性分子
    MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
    MergePops_uni = MergePops[indices]
    MergeCV_uni = MergeCV[indices]
    Mergesmiles_uni = Mergesmiles[indices]
    #合并种群中可行解数量，及位置
    A_num = sum(MergeCV_uni==0)
    ID = np.where(MergeCV_uni==0)[0]
    if A_num==nPop:
        Archive_pops = MergePops_uni[ID]
        Archive_fits = MergeFits_uni[ID]
        Archive_CV = MergeCV_uni[ID]
        Arc_smiles = Mergesmiles_uni[ID]
    elif A_num<nPop:
        # 将 a, b, c 数组按照 CV 的值排序
        sorted_indices = np.argsort(MergeCV_uni, axis=0)
        Archive_pops = MergePops_uni[sorted_indices.ravel()[0:nPop], :]
        Archive_fits = MergeFits_uni[sorted_indices.ravel()[0:nPop], :]
        Archive_CV = MergeCV_uni[sorted_indices.ravel()[0:nPop]]
        Arc_smiles = Mergesmiles_uni[sorted_indices.ravel()[0:nPop]]

    else:
        # 可行解按照支配排序和拥挤度筛选
        Archive_pops,Archive_fits = archive_nsga2(MergePops_uni[ID],MergeFits_uni[ID], nPop)
        Archive_CV = MergeCV_uni[ID]
        Archive_CV = Archive_CV[0:nPop]
        Arc_smiles = Mergesmiles_uni[ID]
        Arc_smiles = Arc_smiles[0:nPop]

    return Archive_pops,Archive_fits, Archive_CV, Arc_smiles


'''
nPop=10
nChr = 5
nF = 2
newPops = np.zeros((nPop, nChr))
newFits = np.zeros((nPop, nF))
MergeFits_uni= np.random.rand(20, 2)
MergePops_uni = np.random.rand(20, 5)
MergeRanks = nonDominationSort(MergePops_uni, MergeFits_uni)
MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni, MergeRanks)

indices = np.arange(MergePops_uni.shape[0])
r = 0
i = 0
rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
while i + len(rIndices) < nPop:
    newPops[i:i + len(rIndices)] = MergePops_uni[rIndices]
    newFits[i:i + len(rIndices)] = MergeFits_uni[rIndices]
    r += 1  # 当前等级+1
    i += len(rIndices)
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    #若加到最后一个等级仍不足种群数，随机采分子
    #不够的分子采样补全
    if r == max(MergeRanks)+1:
        IID = indices.tolist()
        for j in range(i,nPop):
            idx1, idx2 = random.sample(IID, 2)  # 随机挑选两个个体
            idx = compare(idx1, idx2, MergeRanks, MergeDistances)
            newPops[j] = MergePops_uni[idx]
            newFits[j] = MergeFits_uni[idx]
            j += 1
            i += 1

if i < nPop:
    rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
    rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
    surIndices = rIndices[rSortedIdx[:(nPop - i)]]
    newPops[i:] = MergePops_uni[surIndices]
    newFits[i:] = MergeFits_uni[surIndices]
'''