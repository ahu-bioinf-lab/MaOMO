import argparse
import os

import numpy as np
from tensorboardX import SummaryWriter
# import tensorflow as tf
import time

from sub_code_many.models import CDDDModel
from sub_code_many.NSGA2many import *
from sub_code_many.property import *
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
import pygmo as pg
import math

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", default='obj5',
                        choices=['qed', 'logP', 'qedplogp', 'obj5'])
    args = parser.parse_args()
    if args.opt == 'qed':
        ff = ff_qed
        fitness = fitness_qed
        col = ['SMILES', 'mol_id', 'qed', 'sim']
    elif args.opt == 'logP':
        ff = ff_plogp
        fitness = fitness_plogp
    elif args.opt == 'qedplogp':
        ff = ff_qedlogp
        fitness = fitness_qedlogp
        col = ['SMILES', 'mol_id', 'qed', 'sim', 'plogp_imp']
    elif args.opt == 'obj5':
        ff = ff_obj5
        fitness = fitness_obj5
        col = ['SMILES', 'mol_id','restart', 'qed', 'gsk3b', 'jnk', 'sa_nom', 'sim']

    args = parser.parse_args()
    # device = 'cuda'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    ######加载预训练的JTVAE
    model = CDDDModel()
    # canonicalize
    data = pd.read_csv('./data/gskjnk_test.csv').values

    simbank = pd.read_csv('./data/bank library/gskjnk_bank.csv').values
    smi_end_all = []  # save molecules in the last generation
    HV_all = []  # save HV at each generation
    diversity_all = []  # save diversity
    sr_pop_all = []  # save the number of successful molecules
    adap_pro_all = []  # save the selected key property
    runtime = []  # save running time
    num_rank_all = []  # save the number of non-dominated molecules
    # paremeters
    nPop = 100  # population size
    pc = 1  # crossover rate
    pm = 0.5  # mutation rate
    nChr = 512  # length of latent vector
    lb = -1  # lower bounder
    rb = 1  # upper bounder
    numberOfGroups = 16  # number of fragments
    d = 0.26  # crossover parameter
    disM = 20
    nIter = 100  # total generation
    restart = 1
    adap_tres = np.array([0.7, 0.4, 0.4, 0.7, 0.2])#用于计算属性大于阈值个数，固定不变
    L = 20  # level based dominance
    SR = 0
    #a = list(range(0, 481, 20))  # 180
    #b = list(range(420, 501, 20))  # 200
    a = [0]
    b = [1]

    # T_eval = []
    # time_all = []

    for num in range(len(a)):
        mm1 = a[num]
        mm2 = b[num]
        fits_pop_all = []  # 储存每个分子每一代种群属性值，（Mol*Npop）*(T*M)，分批存
        smiles_all = []  # 储存每个分子每一代种群分子SMILES，（Mol*T）*Npop，分批存
        for i in range(mm1,mm2):
            #load each lead molecule
            nn = i
            # molecular SMILES
            smiles = data[i][0]
            print('ori_smiles:', smiles)
            mol_0 = Chem.MolFromSmiles(smiles)
            seq = Chem.MolToSmiles(mol_0)
            z_0 = model.encode(smiles)  # 对分子序列进行编码
            fp_0 = morgan_fingerprint(Chem.MolFromSmiles(seq))  # 分子序列的摩根指纹，用于计算相似性
            fits_0 = ff(seq, mol_0, fp_0)
            print(fits_0)
            # embedding of molecules in the bank library
            aa = simbank[:, 1] == i
            index = np.where(aa == 1)
            simbank_i = simbank[index]
            num_int = len(simbank_i)
            bank_emb = np.zeros((num_int, 512))
            for i in range(num_int):
                bank_emb[i] = model.encode(simbank_i[i][0])
            bank_emb = torch.tensor(bank_emb).to(torch.float32)

            hv_pop = np.zeros(nIter)
            diversity_iter = np.zeros(nIter)
            adap_fits = np.zeros(nIter)
            num_pop_sr = np.zeros(nIter)
            fits_pop_iter = np.zeros(nPop * 5 )
            num_rank_iter = np.zeros(nIter * 10 )
            #start evolution
            r = 0
            while r < restart:
                t1 = time.time()
                # generate initial populaiton
                bankpop = crossover_z0(z_0, bank_emb, pc, d, lb, rb)
                bankpop = torch.tensor(bankpop).to(torch.float32)
                bankmol, banksmiles = model.decode(bankpop)
                banksmiles = np.array(banksmiles)
                bankfits = fitness(banksmiles, bankmol, fp_0)  # 适应度计算
                bankfits[np.isnan(bankfits)] = 0
                pops, fits, smis = select1_level(nPop, bankpop, bankfits, banksmiles, L)#select1_level
                iter = 0
                while iter < nIter:
                    print("【进度】【{0:20s}】【正在进行{1}代...】【共{2}代】". \
                          format('▋' * int(iter / nIter * 20), iter, nIter), end='\r')
                    ##crossover
                    pr = iter/nIter*(0.5)
                    chrpops, fit_adap = crossover_rev(bank_emb, pops, fits, pc, d, lb, rb, adap_tres, pr)  # 混合线性交叉
                    adap_fits[iter] = fit_adap
                    #mutation
                    chrpops = group_mut(chrpops, lb, rb, pm, numberOfGroups)
                    # evaluation
                    chrpops = torch.from_numpy(chrpops)
                    chrmol, chrsmiles = model.decode(chrpops)
                    chrfits = fitness(chrsmiles, chrmol, fp_0)
                    chrfits[np.isnan(chrfits)] = 0
                    # selection, update population
                    tres = np.array([0.6, 0.2, 0.2, 0.7,0.1])*iter/nIter
                    alpha_diversity = math.cos(iter / nIter * math.pi)
                    pops, fits, smis, pareto_rank = optSelect_manyNSGA2(pops, fits,smis, chrpops, chrfits, chrsmiles, nPop, tres, L, alpha_diversity)
                    num_rank_iter[10*iter:10*(iter+1)] = pareto_rank
                    #save properties of molecules
                    rr_pop = 0
                    unique_smiles = []
                    for i in range(len(smis)):
                        if smis[i] not in unique_smiles:
                            unique_smiles.append(smis[i])
                            fits_pop_iter[i * 5 + 0]= fits[i][0]
                            fits_pop_iter[i * 5 + 1] = fits[i][1]
                            fits_pop_iter[i * 5 + 2] = fits[i][2]
                            fits_pop_iter[i * 5 + 3] = fits[i][3]
                            fits_pop_iter[i * 5 + 4] = fits[i][4]
                            if (np.array(fits[:][i]) >= np.array([0.7, 0.4, 0.4, 0.7,0.2])).all() == 1:
                                rr_pop = rr_pop + 1
                    num_pop_sr[iter] = rr_pop
                    #save molecular SMILES
                    smiles_all.append(smis)
                    fits_pop_all.append(fits_pop_iter)

                    ###calulate and save the HV at each iter
                    try:
                        dominated_hypervolume = pg.hypervolume(np.array(
                            [[-1, -1, -1, -1,-1] * np.array(fit) for
                             fit in fits if
                             (np.array(fit) > [0, 0, 0, 0,0]).all()])).compute(np.zeros(5))
                    except:
                        dominated_hypervolume = 0
                    hv_pop[iter] = dominated_hypervolume

                    ### calculate and save the molecular structure diversity at each iter
                    mol = [Chem.MolFromSmiles(seq) for seq in unique_smiles]
                    fp = [morgan_fingerprint(m) for m in mol]
                    div_mean = np.zeros((len(mol), len(mol)))
                    for i in range(len(mol)):
                        fp_111 = fp[i]
                        for j in range(i + 1, len(mol)):
                            try:
                                div_mean[i][j] = tanimoto_similarity(mol[j], fp_111)
                            except:
                                div_mean[i][j] = 1
                    if len(mol)>1:
                        mean_diversity = 1 - sum(sum(div_mean)) * 2 * (1 / (len(mol) * (len(mol) - 1)))
                    else:
                        mean_diversity = 1
                    #print('mean structure diversity：', mean_diversity)
                    diversity_iter[iter] = mean_diversity

                    iter = iter + 1
                    print(iter)
                # save the optimzied molecules in the last generation
                num_rank_all.append(num_rank_iter)
                endsmiles = np.array(smis)
                endmol = []
                for i in range(len(endsmiles)):
                    endmol.append(Chem.MolFromSmiles(endsmiles[i]))
                endmol = np.array(endmol)

                endfits = fitness(endsmiles, endmol, fp_0)
                rr = []
                unique_smiles = []
                for i in range(len(endsmiles)):
                    if endsmiles[i] not in unique_smiles:
                        unique_smiles.append(endsmiles[i])
                        tuple = (endsmiles[i], nn,r, endfits[i][0], endfits[i][1], endfits[i][2], endfits[i][3], endfits[i][4])
                        smi_end_all.append(tuple)
                        if (np.array(endfits[:][i]) >= np.array([0.7, 0.4, 0.4,0.7,0.2])).all() == 1:
                            rr.append(1)

                if 1 in rr:
                    SR = SR + 1
                r = r + 1
                print('restart:', r)
                t2 = time.time()
                time_1 = (t2 - t1) / 60
                print('run time:', time_1)
                runtime.append(time_1)

            HV_all.append(hv_pop)
            diversity_all.append(diversity_iter)
            sr_pop_all.append(num_pop_sr)
            adap_pro_all.append(adap_fits)
            result = [nn - a[0] + 1, SR]
            print('result-all,SR:', result)
            print('save mol:', nn)

            task = 'task3'
            name = 'MaOMO'
            np.savetxt('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_HV_pop.txt', HV_all,
                       fmt='%s')  # save HV
            np.savetxt('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_diversity_pop.txt', diversity_all,
                       fmt='%s')  # save diversity
            np.savetxt('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_num_rank.txt', num_rank_all,
                       fmt='%d')  # save number of non-dominated molecules
            np.savetxt('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_sr_pop.txt', sr_pop_all,
                       fmt='%d')  # save number of successfully optimized molecules
            np.savetxt('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_adapobj.txt', adap_pro_all,
                       fmt='%4f')  # save key property ID
            df_to_save_A_to_B = pd.DataFrame(smi_end_all, columns=col)
            df_to_save_A_to_B.to_csv('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_endsmiles.csv',
                                     index=False)  # save optimized molecules
            np.savetxt('./results/' + task + '/' + name + str(a[0]) + str(b[-1]) + '_runtime.txt', runtime,
                       fmt='%s')  # save running time

            np.savetxt('./results/' + task + '/' + name + str(mm1) + str(mm2) + '_smiles_iter.txt', smiles_all,
                       fmt='%s')  # save molecules in all generation
            np.savetxt('./results/' + task + '/' + name + str(mm1) + str(mm2) + '_fits_pop.txt', fits_pop_all,
                       fmt='%4f')  # save properties in all generation



# writer.close()
