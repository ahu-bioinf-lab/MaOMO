# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 12:45:28 2022

@author: 86136
"""

from functools import lru_cache
import torch
import numpy as np
import pandas as pd
import rdkit
from moses.metrics import QED as QED_
from moses.metrics import SA, logP
from rdkit.Chem import AllChem
#from DRD2.DRD2_predictor2 import *
#from drd.drd_model import *
from tdc import Oracle
from rdkit.Chem import Descriptors

from rdkit import Chem, DataStructs


def penalized_logP(mol):
    """Penalized logP.

    Computed as logP(mol) - SA(mol) as in JT-VAE.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: Penalized logP or NaN if mol is None.
    """
    #mol = Chem.MolFromSmiles(mol)
    try:
        return logP(mol) - SA(mol)
    except:
        return np.nan


def QED(mol):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    #mol = Chem.MolFromSmiles(mol)
    try:
        return QED_(mol)
    except:
        return np.nan
    
def morgan_fingerprint(mol):
    """Molecular fingerprint using Morgan algorithm.

    Uses ``radius=2, nBits=2048``.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the fingerprint.

    Returns:
        np.ndarray: Fingerprint vector.
    """
    #mol = Chem.MolFromSmiles(seq)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def tanimoto_similarity(seq, fp_0):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    #mol = Chem.MolFromSmiles(seq)
    fp = morgan_fingerprint(seq)
    try:
        return rdkit.DataStructs.TanimotoSimilarity(fp_0, fp)
    except:
        return np.nan


def sim_2(mol1, mol2):
    """Tanimoto similarity between two molecules.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.
        fp_0 (array): Fingerprint vector of original molecule.

    Returns:
        float: Tanimoto similarity.
    """
    #mol = Chem.MolFromSmiles(mol)
    fp_1 = morgan_fingerprint(mol1)
    fp_2 = morgan_fingerprint(mol2)
    if fp_1 is None:
        return 0
    return rdkit.DataStructs.TanimotoSimilarity(fp_1, fp_2)

'''
drd2_model = drd2_model()
def cal_DRD2(molecule_SMILES):
    return drd2_model(molecule_SMILES)
'''
def cal_SA(mol):
    #mol = Chem.MolFromSmiles(mol)
    try:
        return SA(mol)
    except:
        return np.nan


qed_ = Oracle('qed')
sa_ = Oracle('sa')
jnk_ = Oracle('JNK3')
gsk_ = Oracle('GSK3B')
logp_ = Oracle('logp')
drd2_ = Oracle('drd2')
fexofenadine_ = Oracle('fexofenadine_mpo')
osimertinib_ = Oracle('osimertinib_mpo')
def normalize_sa(smiles):
    try:
        sa_score = sa_(smiles)
        normalized_sa = (10. - sa_score) / 9.
        return normalized_sa
    except:
        return np.nan

def jnk(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return jnk_(smi)
    except:
        return np.nan



def gsk(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return gsk_(smi)
    except:
        return np.nan

def drd2(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return drd2_(smi)
    except:
        return np.nan

smarts = pd.read_csv(('./data/archive/sure_chembl_alerts.txt'), header=None, sep='\t')[1].tolist()
alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]
def cv(mol, c):
    try:
        ri = mol.GetRingInfo()#分子中的环
        len_ring = [len(r) for r in ri.AtomRings()]#多个环的原子数
        #计算环原子数的违反度
        if len(len_ring)==0:
            cv_ring = 0
        else:
            cv_ring = [max([x-6,0])+max([5-x,0]) for x in len_ring]
            cv_ring = sum(cv_ring)
        #计算分子量的违反度
        if c==2:
            #wt = Descriptors.MolWt(mol)  # 分子量
            #cv_wt = max([wt-500,0])
            ls_tox = [mol.HasSubstructMatch(alert) for alert in alert_mols]
            ls_tox = np.array(ls_tox)
            cv_tox = np.sum(ls_tox != 0)
            return cv_ring, cv_tox
        else:
            return cv_ring
    except:
        if c==2:
            return 100, 10000
        else:
            return 100

def fexo(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return fexofenadine_(smi)
    except:
        return np.nan

def osim(smi):
    """Drug like-ness measure.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Molecule on which to compute the metric.

    Returns:
        float: QED or NaN if mol is None.
    """
    try:
        return osimertinib_(smi)
    except:
        return np.nan

#计算相同区域支配等级中分子的结构多样性值，越大越好
def cal_div(smiles, ranks):
    #pen_logP = penalized_logP(seq)#需改为种群
    num = smiles.shape[0]#分子个数
    fp = [morgan_fingerprint(Chem.MolFromSmiles(seq)) for seq in smiles]
    mol = [Chem.MolFromSmiles(seq) for seq in smiles]

    uni_level = sorted(list(set(ranks)))  # 从小到大
    indices = np.arange(num)

    div = np.zeros(num)

    for r in uni_level:
        rIdices = indices[ranks==r]  # 当前等级种群的索引
        #只计算分子与相同等级的分子结构多样性
        for i in rIdices:
            fp_0 = fp[i]
            sim=[]
            for j in rIdices:
                try:
                    sim.append(tanimoto_similarity(mol[j], fp_0))
                except:
                    sim.append(0)
            #最相似的两个分子,与自身相似性1排除
            sorted_list = sorted(sim, reverse=True)
            top_two = sorted_list[1:3]
            div[i] = 1-sum(top_two)/2
    return div

#计算目标空间多样性和分子结构多样性的综合得分,无法按照等级排序
def MIX_div(distance, diversity, alpha_diversity):
    #从大到小排序索引为位置
    SortedIdx_dis = np.argsort(distance)[::-1]
    SortedIdx_mol = np.argsort(diversity)[::-1]
    #位置转为分数
    S1 = np.zeros(len(distance))
    S1[SortedIdx_dis] = np.arange(len(distance))
    S2 = np.zeros(len(distance))
    S2[SortedIdx_mol] = np.arange(len(distance))
    #计算混合分数
    S_all = 0.5 * (1 + alpha_diversity) * S1 + (1 - 0.5 * (1 + alpha_diversity)) * S2  # 分数越小越好#
    return S_all

'''
smi='COc1cccc(O[C@@H]2CC[C@H]([NH3+])C2)n1'
smi2 = 'COc1ccc(C(C)=O)c(OCC(=O)N2[C@@H](C)CCC[C@H]2C)c1'
print(qed_(smi))
print(logp_(smi))
mol_0 = Chem.MolFromSmiles(smi)
print(penalized_logP(mol_0))

print(logp_(smi2))
mol_2 = Chem.MolFromSmiles(smi2)
print(penalized_logP(mol_2))

print(jnk(smi))
print(gsk(smi))
print(drd2(smi))
print(fexo(smi))
print(osim(smi))
'''

'''
smi = 'CN1C(=O)CC(N2c3ccccc3SC(c3ccco3)C2c2ccco2)c2ccccc21'#,'CN1C(=O)CC(N2c3ccccc3SC(c3ccco3)C2c2ccco2)c2ccccc21']
test_mol = Chem.MolFromSmiles(smi)
HA = test_mol.GetNumHeavyAtoms()
print(HA)
#cv_mol = cv(test_mol,2)
#print(cv_mol)
'''





