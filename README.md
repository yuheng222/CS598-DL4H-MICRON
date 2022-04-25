# CS598-DL4H-MICRON

## Overview
 
This repository contains code which we used for our reproducibility study project as part of the CS598 - Deep
Learning for Healthcare course at University of Illinois Urbana-Champaign. Our project aims to replicate the
experiments and verify the claims and evaluation results that were presented in the [Change Matters: Medication Change
Prediction with Recurrent Residual Networks (MICRON) paper](https://arxiv.org/abs/2105.01876). 

We referenced most of our code from the [MICRON repository](https://github.com/ycq091044/MICRON). We did some refactoring
on their code, made some modifications to its structure and added more documentation to make it easier for users 
to install and run the code.

## System Requirements
If you are using a fresh Ubuntu VM, the easiest way to run this code base is to use 
[Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software). It will install all the required data 
science libraries and packages for you automatically in a single command. We have tested that our code runs successfully 
with Lambda Stack.

Our code also runs successfully on a machine with the below specifications.

System:
- Ubuntu 20.04.4 LTS
- Python 3.8.10
- CUDA 11.6

Python packages:
- dill==0.3.4
- numpy==1.17.4
- pandas==0.25.3
- scikit-learn==0.22.2.post1
- torch==1.10.1

This code may or may not run successfully on any other versions of Ubuntu, Python and CUDA as we have not
tested the code on other environments. If you are using other versions, please leave a comment in the issues section and 
let us know if you are able or not able to run the code.

## Folder Structure

```
📦CS598-DL4H-MICRON
 ┣ 📂config
 ┃ ┗ 📜config.ini
 ┣ 📂data
 ┃ ┣ 📜DIAGNOSES_ICD.csv
 ┃ ┣ 📜PRESCRIPTIONS.csv
 ┃ ┣ 📜PROCEDURES_ICD.csv
 ┃ ┣ 📜drug-DDI.csv
 ┃ ┣ 📜drug-atc.csv
 ┃ ┣ 📜idx2drug.pkl
 ┃ ┣ 📜ndc2atc_level4.csv
 ┃ ┗ 📜ndc2rxnorm_mapping.txt
 ┣ 📂models
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜layers.py
 ┃ ┗ 📜models.py
 ┣ 📂utils
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┗ 📜util.cpython-38.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜preprocessing.py
 ┃ ┗ 📜util.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜dualnn.py
 ┣ 📜gamenet.py
 ┣ 📜leap.py
 ┣ 📜micron.py
 ┣ 📜requirements.txt
 ┣ 📜retain.py
 ┣ 📜run_all.sh
 ┗ 📜simnn.py
```

The required data files in the */data* folder can be obtained from the following links:
- MIMIC-III dataset from [PhysioNet](https://physionet.org/content/mimiciii/1.4/) 
(DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv)
- Medical code mappings from [GAMENet repository](https://github.com/sjy1203/GAMENet/tree/master/data)
(drug-atc.csv, ndc2atc_level4.csv, ndc2rxnorm_mapping.txt, idx2drug.pkl)
- Drug DDI information from [CID](https://drive.google.com/file/d/1mnPc0O0ztz0fkv3HF-dpmBb8PLWsEoDz/view) (drug-DDI.csv)

## Installation
1. Install [Python 3.8](https://www.python.org/downloads) if it's not already installed.
2. If you are using GPU for training/testing, please ensure that compatible versions of 
[cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda-downloads) are installed.
3. Install the required Python packages using pip.
```
pip install -r requirements.txt
```
4. Ensure that all data and mapping files shown in the folder structure section are present in the *data/* folder and 
run the preprocessing file *utils/preprocessing.py*.
```
python preprocessing.py
```
5. Verify that the configurations in *config/config.ini* are correct before running the models for training/testing.

## Training
To train a model, simply run the model's main function (e.g. micron.py, gamenet.py) located in the root folder.
```
python micron.py
```
Every model has its own set of input arguments. To get the full list of input arguments and their usages, use the 
*--help* argument.
```
python micron.py --help
```
The training results will be stored in the *results/* folder and the trained weights will be stored in the *saved/* 
folder respectively.

If you wish to train all models at once, you can run *run_all.sh*. Make sure that the current user has read and execute 
permissions for this shell script.
```
chmod u+r+x run_all.sh
./run_all.sh
```

## Testing
You need to first configure a valid checkpoint path (e.g. saved/micron/Epoch_39_JA_0.5227_DDI_0.07215.model) for the
model that you want to test under the *resume_path* variable in *config/config.ini*.
```
resume_path=saved/micron/Epoch_39_JA_0.5227_DDI_0.07215.model
```
Then, run the model's main function in test mode using the *--test* input argument.
```
python micron.py --test
```

## Results

![Evaluation Results](https://github.com/yuheng222/CS598-DL4H-MICRON/blob/main/results/images/evaluation_result.png?raw=true)

Our evaluation results are very close to what was presented in the paper. From our experiments, MICRON had a 2.9% 
relative improvement in F1-score over the best baseline model (GAMENet) as compared to the 3.5% relative improvement 
mentioned in the paper.

## Citation
```
@inproceedings{yang2021micron,
    title = {Change Matters: Medication Change Prediction with Recurrent Residual Networks},
    author = {Yang, Chaoqi and Xiao, Cao and Glass, Lucas and Sun, Jimeng},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI} 2021},
    year = {2021}
}
```


