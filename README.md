# coinstrained multi-objective molecule optimization (CMOMO)

Implementation of the method proposed in the paper "Balancing property optimization and constraint
satisfaction for constrained multi-property molecular
optimization" by Xin Xia, Yajie Zhang, Xiangxiang Zeng, Xingyi Zhang, Chunhou Zheng, Yansen Su.<sup>1</sup>

### Dependencies
- [cddd](https://github.com/jrwnter/cddd)
  - Notice: You need download the pre-trained encoder-decoder CDDD model to mapping molecules between SMILES and continuous vectors. It can be load by the bash script:
```
./download_default_model.sh
```
The link is also provided on [cddd](https://drive.google.com/file/d/1ccJEclD1dxTNTUswUvIVygqRQSawUCnJ/view?usp=sharing). 

### Installing
The packages need to install: 
- python=3.6
  - rdkit
  - pytorch=1.4.0
  - cudatoolkit=10.0
  - tensorboardX
  - PyTDC
  - Guacamol
  - pip>=19.1,<20.3
  - pip:
    - molsets
    - cddd
- The installed environment can be downloaded from the cloud drive
  - [qmocddd](https://drive.google.com/file/d/1Wad0hxEfoqC5VzWGDPk9eBsFVkCi2o6Y/view?usp=drive_link)

### Data Description
- data/qedplogp_test: dataset on Task1.
- data/Guacamol_sample_800: dataset on Task2.
- data/docking_test: dataset on Task3.
- data/gsk3_test: dataset on Task4.
- data/archive/: the bank library for each lead molecule on four tasks.

### File Description
- sub_code/fitness.py: The script to calculate the objectives of optimization tasks.
- sub_code/property.py: The script to calculate the molecular properties.
- sub_code/generation_rule.py: The script to generate offspring molecules.
- sub_code/selection_rule.py: The script to compare and select molecules.
- sub_code/models.py: The encoder and decoder process.
- sub_code/calc_no.py: The script to calculate the docking scores.
- sub_code/mechanism.py: Guacamol tasks.
- sub_code/nonDominationSort.py: the non-dominated relationships between molecules.

- download_default_model.sh: download the pre-trained encoder-decoder.
- environment.yml: install the environment.
- CMOMO_task1.py: optimization Task1. 
- CMOMO_task2.py: optimization Task2. 
- CMOMO_task3.py: optimization Task3. 
- CMOMO_task4.py: optimization Task4.

- Results/: final optimized molecules on four tasks obtained by CMOMO.

### Optimization
```
python CMOMO_task1.py
python CMOMO_task2.py
python CMOMO_task3.py
python CMOMO_task4.py
```
The output results, i.e., optimized molecules, are summarized in TSCMO_task1_endsmiles, and further save in .csv file.



### Writing your own Objective Function
The fitness function can wrap any function that has following properties:
- Takes a RDKit mol object as input and returns a number as score.
- Uses pyTDC platform to gets the properties such as QED, logp, Drd2, JNK3,...
- Uses docking platform such as qvina 2 to get the protein-ligand docking score.


## Citation

