# adaptive evolutionary-based many-objective molecular optimization framework (MaOMO)

Implementation of the method proposed in the paper "Leveraging adaptive evolutionary optimization for drug molecular design involving many properties" by Xin Xia, Yajie Zhang, Xiangxiang Zeng, Xingyi Zhang, Chunhou Zheng, Yansen Su.<sup>1</sup>

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
- data/Guacamol_sample_800: dataset on Task1.
- data/gsk3_test: dataset on Task2.
- data/gskjnk_test: dataset on Task3.
- data/bank liabrary/: the bank library for each lead molecule on the three tasks.

### File Description
- sub_code_many/NSGA2many.py: The script to generate and select high-quality molecules.
- sub_code_many/property.py: The script to calculate the molecular properties.
- sub_code_many/models.py: The encoder and decoder process.
- sub_code_many/mechanism.py: Guacamol tasks.
- sub_code_many/nonDominationSort.py: the non-dominated relationships between molecules.

- download_default_model.sh: download the pre-trained encoder-decoder.
- environment.yml: install the environment.
- MaOMO_task1.py: optimization Task1. 
- MaOMO_task2.py: optimization Task2. 
- MaOMO_task3.py: optimization Task3. 

- Results/: final optimized molecules on three tasks obtained by MaOMO.

### Optimization
```
python MaOMO_task1.py
python MaOMO_task2.py
python MaOMO_task3.py
```
The output results, i.e., optimized molecules, are summarized in ..._task1_endsmiles, and further save in .csv file.



### Writing your own Objective Function
The fitness function can wrap any function that has following properties:
- Takes a RDKit mol object as input and returns a number as score.
- Uses pyTDC platform to gets the properties such as QED, logp, Drd2, JNK3,...
- Uses docking platform such as qvina 2 to get the protein-ligand docking score.


## Citation

