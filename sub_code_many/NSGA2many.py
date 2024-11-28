# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:07:42 2022

@author: 86136
"""
from property import *
from nonDominationSort_many import *
import torch
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import calc_no as dock
from collections import Counter
from rdkit.Chem import Descriptors
"""
种群初始化 
"""
#def initPops(seqs):
#    pops = model.encode(seqs)
#    return pops
"""
选择算子 
"""
def select1(pool, pops, fits, ranks, distances):
    # 一对一锦标赛选择 
    # pool: 新生成的种群大小 
    nPop, nChr = pops.shape 
    nF = fits.shape[1] 
    newPops = np.zeros((pool, nChr)) 
    newFits = np.zeros((pool, nF))  

    indices = np.arange(nPop).tolist()
    i = 0 
    while i < pool: 
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体 
        idx = compare(idx1, idx2, ranks, distances)

        newPops[i] = pops[idx] 
        newFits[i] = fits[idx] 
        i += 1 
    return newPops, newFits

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

def manycrossover_2(pops,fits,tres, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = pops.shape[0]
    chrPops = np.zeros((nPop * 2, pops.shape[1]))
    #排序种群

    count_greater_than_1 = np.sum(fits > 1, axis=0)
    # 判断种群中满足各属性阈值分子个数
    m_delta=[]
    for i in range(fits.shape[1]):
        m_delta.append(np.sum(fits[:, i] > tres[i]))
    m = max(m_delta)
    sortedID = np.argsort(fits[:, m])  # 按照分数由小到大的位置索引
    sort_pops = pops[sortedID]

    for i in range(0, nPop/2):
        if np.random.rand() < pc:
            mother1 = sort_pops[np.random.randint(nPop/2)]#内部交叉
            mother2 = sort_pops[np.random.randint(nPop/2)]#内部交叉
            alpha = np.random.rand(4)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            #r1 = (-d) + (1 + 2 * d) * alpha[0]
            r = (-d) + (1 + 2 * d) * alpha
            #r2 = (-d) + (1 + 2 * d) * alpha[1]
            chrPops[4 * i] = sort_pops[i]+r[0]*(mother1-sort_pops[i])  # 混合线性交叉
            chrPops[4 * i + 1] = sort_pops[i]+r[1]*(mother2-sort_pops[i])  # 混合线性交叉
            #外部交叉
            mother3 = sort_pops[np.random.randint(nPop / 2, nPop)]
            mother4 = sort_pops[np.random.randint(nPop / 2, nPop)]
            #r1 = (-d) + (1 + 2 * d) * alpha[0]
            #r2 = (-d) + (1 + 2 * d) * alpha[1]
            chrPops[4 * i+2] = sort_pops[i] + r[2] * (mother3 - sort_pops[i])  # 混合线性交叉
            chrPops[4 * i + 3] = sort_pops[i] + r[3] * (mother4 - sort_pops[i])  # 混合线性交叉
        chrPops[chrPops < lb] = lb
        chrPops[chrPops > rb] = rb
    return chrPops

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

def crossover_z0_2(z_0, archive1_emb, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = archive1_emb.shape[0]
    chrPops = np.zeros((nPop*2, z_0.shape[1]))

    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother = z_0
            [alpha1,alpha2]=np.random.rand(2)#生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1=(-d)+(1+2*d)*alpha1
            r2=(-d)+(1+2*d)*alpha2
            chrPops[2*i] = mother+0.8*r1*(archive1_emb[i]-mother)#混合线性交叉
            chrPops[2*i+1] = mother + 0.8*r2* (archive1_emb[i]-mother)  # 混合线性交叉
    chrPops[chrPops < lb] = lb
    chrPops[chrPops > rb] = rb
    return chrPops

def crossover_co_2(arc, pops, pc,d, lb, rb):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    nPop = arc.shape[0]
    chrPops = np.zeros((nPop * 2, pops.shape[1]))
    for i in range(0, nPop):
        if np.random.rand() < pc:
            mother1 = pops[np.random.randint(nPop)]
            mother2 = pops[np.random.randint(nPop)]
            [alpha1, alpha2] = np.random.rand(2)  # 生成2个随机数#,[alpha1,alpha2]=np.random.rand(2)
            r1 = (-d) + (1 + 2 * d) * alpha1
            r2 = (-d) + (1 + 2 * d) * alpha2
            chrPops[2 * i] = arc[i]+r1*(mother1-arc[i])  # 混合线性交叉
            chrPops[2 * i + 1] = arc[i]+r2*(mother2-arc[i])  # 混合线性交叉
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
            zz = torch.tensor(zz).to(torch.float32)
            #zzz = 4*zz-2#-2到2
            #pos = np.random.randint(0,nChr,1)
            pos = np.random.randint(0, nChr, m)#变异2个位置
            pops[i][pos] = zz
    return pops
#分组线性随机变异
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

#分组线性多项式变异
def group_polyMutation(pops, lb, rb,disM, pm,numberOfGroups):
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
    pops = np.minimum(np.maximum(pops, lb), rb)
    Upper = np.full((nPop, nChr), rb)
    Lower = np.full((nPop, nChr), lb)
    # 种群中对应位置计算变化
    pops[np.where(temp == 1)] = pops[np.where(temp == 1)] + (Upper[np.where(temp == 1)] - Lower[np.where(temp == 1)]) * (
                                             (2 * mu[np.where(temp == 1)] + (1 - 2 * mu[np.where(temp == 1)]) *
                                              (1 - (pops[np.where(temp == 1)] - Lower[np.where(temp == 1)]) / (
                                                          Upper[np.where(temp == 1)] - Lower[np.where(temp == 1)])) ** (
                                                          disM + 1)) ** (1 / (disM + 1)) - 1)
    temp = np.where((Site == 1) & (mu > pm), 1, 0)
    pops[np.where(temp == 1)] = pops[np.where(temp == 1)] + (
                Upper[np.where(temp == 1)] - Lower[np.where(temp == 1)]) * (
                                             1 - (2 * (1 - mu[np.where(temp == 1)]) + 2 * (
                                                 mu[np.where(temp == 1)] - 0.5) *
                                                  (1 - (Upper[np.where(temp == 1)] - pops[np.where(temp == 1)]) / (
                                                              Upper[np.where(temp == 1)] - Lower[
                                                          np.where(temp == 1)])) ** (disM + 1)) ** (1 / (disM + 1)))
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


def crossover_rev(intpops, pops,fits, pc,d, lb, rb,tres, pr):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    # 动态参数引入初始种群基因，以动态概率与初始种群个体交叉
    nPop = pops.shape[0]
    #属性值大于等于阈值为1，小于为0
    state = (fits >= tres).astype(int)
    #选出0最多的属性
    nom_state = np.sum(state == 0, axis=0)
    fit_adap = np.argmax(nom_state)#不满足阈值分子最多的属性索引
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
    return chrPops, fit_adap

def crossover_noadap(intpops, pops, pc,d, lb, rb, pr):
    # 拷贝父代种群，以防止改变父代种群结构  ,d=0.25参数
    # 动态参数引入初始种群基因，以动态概率与初始种群个体交叉
    nPop = pops.shape[0]
    chrPops = pops
    #种群的每个分子与前P/2个该属性值高的分子交叉
    for i in range(0, nPop):
        if np.random.rand() < pc:
            if np.random.rand()< pr:
                #parent = intpops[np.random.randint(intpops.shape[0])]
                parent = intpops[np.random.randint(intpops.shape[0])].numpy()
                alpha1 = np.random.rand()  # 生成-1到1的随机数(np.random.rand()-0.5)*2
                r1 = (-d) + (1 + 2 * d) * alpha1
                # r2=(-d)+(1+2*d)*alpha2
                chrPops[i] = chrPops[i] + r1 * (parent - chrPops[i])  # 混合线性交叉
                chrPops[i][chrPops[i] < lb] = lb
                chrPops[i][chrPops[i] > rb] = rb
            else:
                parent = pops[np.random.randint(nPop)]
                alpha1=np.random.rand()#生成-1到1的随机数(np.random.rand()-0.5)*2
                r1=(-d)+(1+2*d)*alpha1
                #r2=(-d)+(1+2*d)*alpha2
                chrPops[i] = chrPops[i]+r1*(parent-chrPops[i])#混合线性交叉
                chrPops[i][chrPops[i]<lb] = lb
                chrPops[i][chrPops[i]>rb] = rb
    return chrPops

def crossover_norev(pops,fits, pc,d, lb, rb,tres):
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
    return chrPops, fit_adap

"""扰动产生新的子代 
多点
"""
def Disturb(pops, nChr,m,lb, rb):
    '''
    nPop = pops.shape[0]
    dis_pop = np.zeros((nPop, nChr))
    gauss = np.random.normal(0, 1, (nPop, nChr))
    dis_pop[:nPop] = gauss*0.5 + pops
    '''
    nPop = pops.shape[0]
    for i in range(nPop):
        pos = np.random.randint(0, nChr, m)  # 变异m个位置
        gauss = np.random.normal(0, 1, m)
        pops[i][pos] = pops[i][pos]+gauss#*0.5
        pops[i][pops[i] < lb] = lb
        pops[i][pops[i] > rb] = rb
    return pops

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
    lde4 = -cal_4lde(seq)#返回正数，越大越好
    print('lde4:',lde4)
    return qed, sim, lde4#pen_logP,

def cal_4lde(seq):
    mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return 10 ** 4
    else:
        lde4 = dock.perform_calc_single(seq, '4lde', docking_program='qvina')
        return lde4



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


"""
种群的合并和优选 
"""

#种群的CV需更新,种群SMILES需更新
def optSelect_uni(pops, fits, CV_pops, smiles, chrPops, chrFits, CV_offspring, chrsmiles, nPop):
    """种群合并与优选
    Return:
        newPops, newFits
    """
    nChr = pops.shape[1]
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    newCV = [0]*nPop
    newsmiles = [0] * nPop
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    MergeCV = np.concatenate((CV_pops, CV_offspring), axis=0)
    Mergesmiles = np.concatenate((smiles, chrsmiles), axis=0)
    #首先去除重复
    MergeFits_uni, indices = np.unique(MergeFits, axis=0, return_index=True)
    MergePops_uni = MergePops[indices]
    MergeCV_uni = MergeCV[indices]
    Mergesmiles_uni = Mergesmiles[indices]
    MergeRanks = nonDominationSort(MergePops_uni, MergeFits_uni)
    MergeDistances = crowdingDistanceSort(MergePops_uni, MergeFits_uni, MergeRanks)

    indices = np.arange(MergePops_uni.shape[0])
    r = 0
    i = 0
    rIndices = indices[MergeRanks == r]  # 当前等级为r的索引
    while i + len(rIndices) < nPop:
        newPops[i:i + len(rIndices)] = MergePops_uni[rIndices]
        newFits[i:i + len(rIndices)] = MergeFits_uni[rIndices]
        newCV[i:i + len(rIndices)] = MergeCV_uni[rIndices]
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
                newCV[j] = MergeCV_uni[idx]
                newsmiles[j] = Mergesmiles_uni[idx]
                j += 1
                i += 1

    if i < nPop:
        rDistances = MergeDistances[rIndices]  # 当前等级个体的拥挤度
        rSortedIdx = np.argsort(rDistances)[::-1]  # 按照距离排序 由大到小
        surIndices = rIndices[rSortedIdx[:(nPop - i)]]
        newPops[i:] = MergePops_uni[surIndices]
        newFits[i:] = MergeFits_uni[surIndices]
        newCV[i:] = MergeCV_uni[surIndices]
        newsmiles[i:] = Mergesmiles_uni[surIndices]
    return (newPops, newFits, newCV, newsmiles)



#种群的CV需更新,种群SMILES需更新
def optSelect_manyNSGA2(pops, fits,smiles, chrPops, chrFits, chrsmiles, nPop, tres_d, L, alpha_diversity):
    """种群合并与优选
    tres:阈值
    iter：当前代数
    Return:
        newPops, newFits
    """
    #计算动态阈值
    #tres_d = tres*iter/T
    nChr = pops.shape[1]
    nF = fits.shape[1]
    newPops = np.zeros((nPop, nChr))
    newFits = np.zeros((nPop, nF))
    #newCV = [0]*nPop
    newsmiles = [0] * nPop
    # 合并父代种群和子代种群构成一个新种群
    MergePops = np.concatenate((pops, chrPops), axis=0)
    MergeFits = np.concatenate((fits, chrFits), axis=0)
    #MergeCV = np.concatenate((CV_pops, CV_offspring), axis=0)
    Mergesmiles = np.concatenate((smiles, chrsmiles), axis=0)
    #首先去除重复
    MergeFits, indices = np.unique(MergeFits, axis=0, return_index=True)
    #去除小于0的行
    #negative_rows = np.where((MergeFits < 0).any(axis=1))[0]
    #MergeFits = np.delete(MergeFits, negative_rows, axis=0)
    #indices = np.delete(indices, negative_rows, axis=0)
    MergePops = MergePops[indices]
    #MergeCV_uni = MergeCV[indices]
    Mergesmiles = Mergesmiles[indices]
    #父代子代分子满足阈值的个数
    n_sr=[]#存储满足阈值的ID
    for i in range(len(MergeFits)):
        if (np.array(MergeFits[:][i])>=np.array(tres_d)).all()== 1:
            n_sr.append(i)
    print('number of molecules satisfy the tres:', len(n_sr))

    if len(n_sr)<5:
        n_sr = np.array(range(len(MergeFits)))
    #满足阈值的解大于种群分子数,从中选择P个
    MergePops_sr = MergePops[n_sr]
    MergeFits_sr = MergeFits[n_sr]
    # MergeCV_uni = MergeCV[n_sr]
    Mergesmiles_sr = Mergesmiles[n_sr]

    #先pareto等级，第一等级继续划分level
    MergeRanks_sr = nonDominationSort(MergePops_sr, MergeFits_sr)
    MergeLevel_sr = level_p(MergeFits_sr, L)#level不是从0开始按顺序
    MergeDistances_sr = crowding_level(MergePops_sr, MergeFits_sr, MergeLevel_sr)

    #保存pareto前沿按等级pareto划分的个数
    pareto_rank = np.zeros(10)
    indices = np.arange(MergePops_sr.shape[0])
    rIndices = indices[MergeRanks_sr == 0]
    cal_level = level_p(MergeFits_sr[rIndices], L)
    uni_l = sorted(list(set(cal_level)))  # 哪些区域等级内有分子，从小到大排序
    count = Counter(cal_level)  # 每个元素出现次数
    for i in range(len(uni_l)):
        if i<9:
            pareto_rank[i] = count[uni_l[i]]  # 对应0-9出现次数
    pareto_rank[9] = len(Mergesmiles)  # 最后一列存去重后整体数量

    if len(n_sr) == nPop:
        newPops = MergePops_sr
        newFits = MergeFits_sr
        # newCV[0:len(n_sr)] = MergeCV_uni[n_sr]
        newsmiles = Mergesmiles_sr
    elif len(n_sr)< nPop:
        newPops[0:len(n_sr)] = MergePops_sr
        newFits[0:len(n_sr)] = MergeFits_sr
        # newCV[0:len(n_sr)] = MergeCV_uni[n_sr]
        newsmiles[0:len(n_sr)] = Mergesmiles_sr
        IID = list(range(len(n_sr)))  # list(range(n_sr))
        for j in range(len(n_sr), nPop):
            idx1, idx2 = random.sample(IID, 2)  # 随机挑选两个个体
            idx = compare(idx1, idx2, MergeLevel_sr, MergeDistances_sr)
            newPops[j] = MergePops_sr[idx]
            newFits[j] = MergeFits_sr[idx]
            newsmiles[j] = Mergesmiles_sr[idx]
            j += 1
    else:
        #pareto支配
        #优先pareto，没超过种群大小的等级分子放入种群
        r = 0
        i = 0
        #rIndices = indices[MergeRanks_sr == r]  # 当前等级为r的索引
        while i + len(rIndices) < nPop:
            newPops[i:i + len(rIndices)] = MergePops_sr[rIndices]
            newFits[i:i + len(rIndices)] = MergeFits_sr[rIndices]
            newsmiles[i:i + len(rIndices)] = Mergesmiles_sr[rIndices]
            r += 1  # 当前等级+1
            i += len(rIndices)
            rIndices = indices[MergeRanks_sr == r]  # 当前等级为r的索引
        if i < nPop:
            MergePops_level = MergePops_sr[rIndices]
            MergeFits_level = MergeFits_sr[rIndices]
            # MergeCV_level = MergeCV_uni[rIndices]
            Mergesmiles_level = Mergesmiles_sr[rIndices]
            cal_level = level_p(MergeFits_level, L)  # 计算分子区域等级
            # 对应目标空间多样性和计算分子结构多样性值,以及混合多样性
            cal_distance = MergeDistances_sr[rIndices]
            cal_diversity = cal_div(Mergesmiles_level, cal_level)
            mix_diversity = MIX_div(cal_distance, cal_diversity, alpha_diversity)
            # 优先pareto，没超过种群大小的等级分子放入种群
            indices = np.arange(MergePops_level.shape[0])
            uni_l = sorted(list(set(cal_level)))  # 哪些区域等级内有分子，从小到大排序
            l = 0
            rIndices = indices[cal_level == uni_l[l]]  # 当前等级为r的索引
            while i + len(rIndices) < nPop:
                newPops[i:i + len(rIndices)] = MergePops_level[rIndices]
                newFits[i:i + len(rIndices)] = MergeFits_level[rIndices]
                # newCV[i:i + len(rIndices)] = MergeCV_level[rIndices]
                newsmiles[i:i + len(rIndices)] = Mergesmiles_level[rIndices]
                # pareto_rank[r+l-1] = count[l]
                l += 1  # 当前等级+1
                i += len(rIndices)
                rIndices = indices[cal_level == uni_l[l]]  # 当前等级为l的索引
            if i < nPop:
                rDistances = mix_diversity[rIndices]  # 当前等级个体的混合多样性
                rSortedIdx = np.argsort(rDistances) # 按照距离排序 越小越好
                surIndices = rIndices[rSortedIdx[:(nPop - i)]]
                newPops[i:] = MergePops_level[surIndices]
                newFits[i:] = MergeFits_level[surIndices]
                newsmiles[i:] = Mergesmiles_level[surIndices]
    return (newPops, newFits, newsmiles, pareto_rank)#newCV,

def select1_level(pool, pops, fits, smiles,L):
    # 一对一锦标赛选择
    # pool: 新生成的种群大小
    fits_uni, indices = np.unique(fits, axis=0, return_index=True)
    pops_uni = pops[indices]
    smi_uni = smiles[indices]
    ranks = nonDominationSort(pops_uni, fits_uni)
    level = level_p(fits_uni, L)  # 计算分子区域等级
    # 计算分子结构多样性值
    diversity = cal_div(smi_uni, level)
    nPop, nChr = pops_uni.shape
    nF = fits_uni.shape[1]
    newPops = np.zeros((pool, nChr))
    newFits = np.zeros((pool, nF))
    newsmiles = [0] * pool
    indices = np.arange(nPop).tolist()
    i = 0
    while i < pool:
        idx1, idx2 = random.sample(indices, 2)  # 随机挑选两个个体
        idx = compare_level(idx1, idx2, ranks, level, diversity)

        newPops[i] = pops_uni[idx]
        newFits[i] = fits_uni[idx]
        newsmiles[i] = smi_uni[idx]
        i += 1
    return newPops, newFits, newsmiles

def compare_level(idx1, idx2, ranks, level, diversity):
    # return: 更优的 idx
    if ranks[idx1] < ranks[idx2]:
        idx = idx1
    elif ranks[idx1] > ranks[idx2]:
        idx = idx2
    else:
        if level[idx1] < level[idx2]:#区域等级越小越好
            idx = idx1
        elif level[idx1] > level[idx2]:
            idx = idx2
        else:
            if diversity[idx1] <= diversity[idx2]:#多样性越大越好
                idx = idx2
            else:
                idx = idx1
    return idx



'''
smi = ['C=c1c2c(nc3c1cc(C(=O)N1COCC1(C)C)n3C)C(CN)=CC=2','CN1C(=O)CC(N2c3ccccc3SC(c3ccco3)C2c2ccco2)c2ccccc21']
test_mol = [Chem.MolFromSmiles(x) for x in smi]
cv_mol = CV(test_mol)
#print(Descriptors.MolWt(mol))


pops = np.random.rand(3, 6)
new_pops = group_mut(pops, -1, 1, 0.5,3)
'''