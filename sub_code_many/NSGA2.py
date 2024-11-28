# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:42 2022

@author: 86136
"""
from property import *
from nonDominationSort import *
#import calc_no as dock
import torch
from tqdm import tqdm
from rdkit.Chem import Descriptors
from collections import Counter
"""
种群初始化 
"""
def initPops(seqs): 
    pops = model.encode(seqs)  
    return pops 
"""
选择算子 
"""
def select1(pool, pops, fits, ranks, distances, smiles):
    # 一对一锦标赛选择 
    # pool: 新生成的种群大小 
    nPop, nChr = pops.shape 
    nF = fits.shape[1] 
    newPops = np.zeros((pool, nChr)) 
    newFits = np.zeros((pool, nF))  
    newsmiles = [0]*pool
    indices = np.arange(nPop).tolist()
    i = 0 
    while i < pool: 
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体 
        idx = compare(idx1, idx2, ranks, distances) 
        newPops[i] = pops[idx] 
        newFits[i] = fits[idx]
        newsmiles[i] = smiles[idx]
        i += 1 
    return newPops, newFits,  newsmiles


def select1_uni(pool, pops, fits, smiles):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    fits_uni, indices = np.unique(fits, axis=0, return_index=True)
    pops_uni = pops[indices]
    smi_uni = smiles[indices]
    ranks = nonDominationSort(pops_uni, fits_uni)
    distances = crowdingDistanceSort(pops_uni, fits_uni, ranks)
    nPop, nChr = pops_uni.shape
    nF = fits_uni.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))
    newsmiles = [0] * pool
    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare(idx1, idx2, ranks, distances)

        newPops[i] = pops_uni[idx]
        newFits[i] = fits_uni[idx]
        newsmiles[i] = smi_uni[idx]
        i += 1
    return newPops, newFits, newsmiles

def compare_single(idx1, idx2, fits_single):
    # return: 更优的 idx
    if fits_single[idx1] < fits_single[idx2]:
        idx = idx2
    elif fits_single[idx1] > fits_single[idx2]:
        idx = idx1
    else:
        idx = idx1
    return idx


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
"""交叉算子 
混合线性交叉 
"""
def crossover_z0(z_0, archive1_emb, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = archive1_emb
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = z_0
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = mother  + r1*(archive1_emb[i]-mother)#0.8*
    chrPops[i][chrPops[i]<lb] = lb
    chrPops[i][chrPops[i]>rb] = rb
    return chrPops
def crossover(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = pops.copy()  
    nPop = chrPops.shape[0]
    for i in range(0, nPop): 
        if np.random.rand() < pc: 
            mother = chrPops[np.random.randint(nPop)]
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = chrPops[i]+r1*(mother-chrPops[i])#混合线性交叉
            chrPops[i][chrPops[i]<lb] = lb 
            chrPops[i][chrPops[i]>rb] = rb 
    return chrPops

def crossover(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    chrPops = pops
    nPop = chrPops.shape[0]
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = chrPops[np.random.randint(nPop)]
            alpha1=np.random.rand()#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = chrPops[i]+r1*(mother-chrPops[i])#混合线性交叉
            chrPops[i][chrPops[i]<lb] = lb
            chrPops[i][chrPops[i]>rb] = rb
    return chrPops

def crossover_adap(pops,fits, pc,d, lb, rb,tres):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = pops.shape[0]
    #属性值大于等于阈值为1，小于为0
    state = (fits >= tres).astype(int)
    #选出0最多的属性
    nom_state = np.sum(state == 0, axis=0)
    fit_adap = np.argmax(nom_state)
    #按照选出的属性从大到小排序
    pop_adap = pops[np.argsort(fits[:, fit_adap])[::-1]]
    chrPops = pops
    #种群的每个分子与前P/2个该属性值高的分子交叉
    for i in range(0, nPop):
        if np.random.rand() < pc:
            parent = pop_adap[np.random.randint(nPop/2)]
            alpha1=np.random.rand()#生成-1到1的随机数(np.random.rand()-0.5)*2
            r1=(-d)+(1+2*d)*alpha1
            #r2=(-d)+(1+2*d)*alpha2
            chrPops[i] = chrPops[i]+r1*(parent-chrPops[i])#混合线性交叉
            chrPops[i][chrPops[i]<lb] = lb
            chrPops[i][chrPops[i]>rb] = rb
    return chrPops, nom_state

def crossover_rev(intpops, pops,fits, pc,d, lb, rb,tres, pr):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    # 动态参数引入初始种群基因，以动态概率与初始种群个体交叉
    nPop = pops.shape[0]
    #属性值大于等于阈值为1，小于为0
    state = (fits >= tres).astype(int)
    #选出0最多的属性
    nom_state = np.sum(state == 0, axis=0)
    fit_adap = np.argmax(nom_state)
    #按照选出的属性从大到小排序
    pop_adap = pops[np.argsort(fits[:, fit_adap])[::-1]]
    chrPops = pops
    #种群的每个分子与前P/2个该属性值高的分子交叉
    for i in range(0, nPop):
        if np.random.rand() < pc:
            if np.random.rand()< pr:
                parent = intpops[np.random.randint(intpops.shape[0])].numpy()
                alpha1 = np.random.rand()  # 生成-1到1的随机数(np.random.rand()-0.5)*2
                r1 = (-d) + (1 + 2 * d) * alpha1
                # r2=(-d)+(1+2*d)*alpha2
                chrPops[i] = chrPops[i] + r1 * (parent - chrPops[i])  # 混合线性交叉
                chrPops[i][chrPops[i] < lb] = lb
                chrPops[i][chrPops[i] > rb] = rb
            else:
                parent = pop_adap[np.random.randint(nPop/2)]
                alpha1=np.random.rand()#生成-1到1的随机数(np.random.rand()-0.5)*2
                r1=(-d)+(1+2*d)*alpha1
                #r2=(-d)+(1+2*d)*alpha2
                chrPops[i] = chrPops[i]+r1*(parent-chrPops[i])#混合线性交叉
                chrPops[i][chrPops[i]<lb] = lb
                chrPops[i][chrPops[i]>rb] = rb
    return chrPops, nom_state

def crossover_2(pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = pops.shape[0]
    chrPops = np.zeros((nPop * 2, pops.shape[1]))
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother1 = pops[np.random.randint(nPop)]
            mother2 = pops[np.random.randint(nPop)]
            [alpha1, alpha2] = np.random.rand(2)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1 = (-d) + (1 + 2 * d) * alpha1
            r2 = (-d) + (1 + 2 * d) * alpha2
            chrPops[2 * i] = pops[i]+r1*(mother1-pops[i])  # 混合线性交叉
            chrPops[2 * i + 1] = pops[i]+r2*(mother2-pops[i])  # 混合线性交叉
        chrPops[chrPops < lb] = lb
        chrPops[chrPops > rb] = rb
    return chrPops

"""变异算子 
单点
"""
def mutate(pops, pm, nChr,m):
    nPop = pops.shape[0] 
    for i in range(nPop):
        if np.random.rand() < pm:
            zz=np.random.rand(m)
            #zzz = 4*zz-2#-2到2
            #pos = np.random.randint(0,nChr,1)
            pos = np.random.randint(0, nChr, m)#变异2个位置
            pops[i][pos] = zz
    return pops

def group_mut(pops, lb, rb, pm,numberOfGroups):
    nPop, nChr = pops.shape
    outIndexarray = creategroup(nPop, nChr, numberOfGroups)
    chosengroups = np.random.randint(1, numberOfGroups + 1, size=outIndexarray.shape[0])  # size=outIndexList.shape[0]
    Site = np.zeros((nPop, nChr))
    for i in range(len(chosengroups)):
        Site[i, :] = (outIndexarray[i, :] == chosengroups[i]).astype(int)
    # 生成随机数判断是否变异
    mu = np.random.rand(nPop, 1)
    mu = np.tile(mu, (1, nChr))
    # 选中的组且小于变异概率，p*emb
    temp = np.where((Site == 1) & (mu < pm), 1, 0)
    pops[np.where(temp == 1)] = (np.random.rand(len(np.where(temp==1)[0]))-0.5)*2
    pops = np.minimum(np.maximum(pops, lb), rb)
    return pops


#生成随机分组索引
def creategroup(nPop,nChr,numberOfGroups):
    outIndexarray = []
    for i in range(nPop):
        # 初始化索引列表
        outIndexList = []
        varsPerGroup = nChr // numberOfGroups
        # 循环生成索引列表
        for i in range(1, numberOfGroups):
            outIndexList.extend(np.ones(varsPerGroup) * i)
        # 补足长度以确保所有变量都被分组
        outIndexList.extend(np.ones(nChr - len(outIndexList)) * numberOfGroups)
        # 对索引列表进行随机排列
        np.random.shuffle(outIndexList)
        outIndexarray.append(outIndexList)
    outIndexarray = np.array(outIndexarray)
    return outIndexarray


"""
种群或个体的适应度 
"""
def fitness_qed(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维 
    nPop = len(mol)
    fits = np.array([ff_qed(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits 

def ff_qed(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim#pen_logP,


def fitness_plogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_plogp(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_plogp(seq, mol, fp_0):
    pen_logP = penalized_logP(mol)#需改为种群
    #qed = QED(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return pen_logP, sim#pen_logP,

##gsk & sim
def fitness_gsk(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_gsk(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_gsk(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    gskb = gsk(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return gskb, sim#pen_logP,

##drd2 & sim
def fitness_drd(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_drd(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_drd(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    drd = drd2(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return drd, sim#pen_logP,

#docking
def fitness_4lde(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)

    fits = np.array([ff_4lde(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_4lde(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    sim = tanimoto_similarity(mol, fp_0)
    lde4 = -cal_4lde(seq)
    return qed, sim, lde4#pen_logP,

def cal_4lde(seq):
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return 10 ** 4
    else:
        lde4 = dock.perform_calc_single(seq, '4lde', docking_program='qvina')
        return lde4

#docking+LE
def fitness_le_1syh(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_le_1syh(seq[i], mol[i], fp_0) for i in range(nPop)])
    return fits

def ff_le_1syh(seq, mol, fp_0):
    sim = tanimoto_similarity(mol, fp_0)
    syh1 = -cal_1syh(seq)
    print('1syh',syh1)
    HA = mol.GetNumHeavyAtoms()
    print('HA', HA)  # 30-60
    if HA < 5:
        HA = 100
    LE = syh1 / HA
    print('LE', LE)
    return sim, LE, syh1, HA#pen_logP,

def cal_1syh(seq):
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return 10 ** 4
    else:
        syh1 = dock.perform_calc_single(seq, '1syh', docking_program='qvina')
        return syh1


##多目标QED,SIM,Plogp
def fitness_qedlogp(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_qedlogp(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits
def ff_qedlogp(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    pen_logP = penalized_logP(mol)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, pen_logP, sim#pen_logP,

def ff_obj2(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    #qed = QED(mol)
    #sa = normalize_sa(seq)
    gskb = gsk(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return gskb, sim#pen_logP,

def fitness_obj2(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_obj2(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_obj3(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    gskb = gsk(seq)
    #gskb = gsk(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, gskb, sim#pen_logP,

def fitness_obj3(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_obj3(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_obj4(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    gskb = gsk(seq)
    sim = tanimoto_similarity(mol, fp_0)
    sa = normalize_sa(seq)
    return qed, gskb, sim, sa#pen_logP,

def fitness_obj4(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_obj4(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_obj5(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    gskb = gsk(seq)
    sim = tanimoto_similarity(mol, fp_0)
    sa = normalize_sa(seq)
    jnk3 = jnk(seq)
    return qed, gskb, jnk3, sa, sim#pen_logP,

def fitness_obj5(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_obj5(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits

def ff_obj6(seq, mol, fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)
    pen_logP = penalized_logP(mol)
    sim = tanimoto_similarity(mol, fp_0)
    sa = normalize_sa(seq)
    gskb = gsk(seq)
    jnk3 = jnk(seq)
    return qed, pen_logP, sim, sa, gskb,jnk3#pen_logP,

def fitness_obj6(seq, mol,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(mol)
    fits = np.array([ff_obj6(seq[i], mol[i], fp_0) for i in range(nPop)])

    return fits



#多目标qed,drd2,sim
def fitness_qeddrd(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qeddrd(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qeddrd(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    drd = drd2(seq)
    #sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim, drd  # pen_logP


#多目标qed,gskb,sa_nom,sim
def fitness_qedgsksa(seqs,mols,fp_0):
    # 计算种群或者个体的适应度，嵌入需解码为SMILES
    # 如果是1维需要转为2维
    nPop = len(seqs)
    fits = np.array([ff_qedgsksa(seqs[i], mols[i], fp_0) for i in range(nPop)])
    return fits

def ff_qedgsksa(seq,mol,fp_0):
    #pen_logP = penalized_logP(seq)#需改为种群
    qed = QED(mol)  # 需改为种群
    gskb = gsk(seq)
    sa_nom = normalize_sa(seq)
    sim = tanimoto_similarity(mol, fp_0)
    return qed, sim, gskb, sa_nom  # pen_logP


"""
种群的合并和优选 
"""
#种群的CV需更新
def optSelect_uni(pops, fits, smiles, chrPops, chrFits, chrsmiles, nPop):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nChr = pops.shape[1]
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    newsmiles = [0] * nPop
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    Mergesmiles = np.concatenate((smiles, chrsmiles), axis=0)
    #首先去除重复
    MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
    MergePops_uni = MergePops[indices]
    Mergesmiles_uni = Mergesmiles[indices]
    MergeRanks = nonDominationSort(MergePops_uni, MergeFits_uni)
    MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni, MergeRanks)

    #帕累托等级存储
    pareto_rank = np.zeros(10)
    count = Counter(MergeRanks)#每个元素出现次数
    for i in range(9):
        pareto_rank[i] = count[i]#对应0-9出现次数
    pareto_rank[9] = len(MergeRanks)#最后一列存去重后整体数量

    indices = np.arange(MergePops_uni.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) < nPop:
        newPops[i:i + len(rIndices)] = MergePops_uni[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits_uni[rIndices]
        newsmiles[i:i + len(rIndices)] = Mergesmiles_uni[rIndices]
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        # 若加到最后一个等级仍不足种群数，随机采分子
        # 不够的分子采样补全
        if r == max(MergeRanks) + 1:
            IID = indices.tolist()
            for j in range(i, nPop):
                idx1, idx2 = random.sample(IID, 2)  # 随机挑选两个个体
                idx = compare(idx1, idx2, MergeRanks, MergeDistances)
                newPops[j] = MergePops_uni[idx]
                newFits[j] = MergeFits_uni[idx]
                newsmiles[j] = Mergesmiles_uni[idx]
                j += 1
                i += 1

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops_uni[surIndices]
        newFits[i:] = MergeFits_uni[surIndices]
        newsmiles[i:] = Mergesmiles_uni[surIndices]
    return (newPops, newFits, newsmiles,pareto_rank)

'''
def optSelect_id(pops, fits, chrPops, chrFits):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)
    optse_id = []
    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) <= nPop:
        newPops[i:i + len(rIndices)] = MergePops[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits[rIndices]
        for j in rIndices:
            optse_id.append(j)
        r += 1  # 当前等级+1
        i += len(rIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
        for s in surIndices:
            optse_id.append(s)
    return newPops, newFits, optse_id

#接收概率，按照NSGA2排序，分子仍有一定概率不被接收
def optSelect_ap(pops, fits, chrPops, chrFits, ap):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nPop, nChr = pops.shape
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeRanks = nonDominationSort(MergePops, MergeFits)
    MergeDistances = crowdingDistanceSort(MergePops, MergeFits, MergeRanks)

    indices = np.arange(MergePops.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    apIndices = rIndices[np.random.rand(len(rIndices)) <= ap]  # 当前等级中通过概率筛选的索引

    while i + len(apIndices) <= nPop:
        newPops[i:i + len(apIndices)] = MergePops[apIndices]#添加满足概率的当前等级的解
        newFits[i:i + len(apIndices)] = MergeFits[apIndices]
        r += 1  # 当前等级+1
        i += len(apIndices)
        rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
        apIndices = rIndices[np.random.rand(len(rIndices)) <= ap]  # 当前等级中通过概率筛选的索引

    if i < nPop:
        rDistances = MergeDistances[apIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops[surIndices]
        newFits[i:] = MergeFits[surIndices]
    return (newPops, newFits)
'''

'''
smi = 'CCCCCC1CCC(CCCCCCCCNCc2ccc([O-])c[nH+]2)CC1'
mol = Chem.MolFromSmiles(smi)
print(Descriptors.MolWt(mol))
'''
