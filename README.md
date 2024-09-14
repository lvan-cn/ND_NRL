# ND_NRL
This repository provides access to the ND_NRL reference model described in the paper "A Name Disambiguation Method Based on Network Representation Learning in Scientific Literature" by Fan Ye, Qing Yang, Zong Xia, Zhili Chen, Hong Zhong, Lu Liu.

## Installation
To install this package, run the following:
```
git clone https://github.com/lvan-cn/ND_NRL
cd ND_NRL
conda create -y --name nd_nrl python==2.7
conda activate nd_nrl
pip install -r requirements.txt
```

## Dataset
Download the dataset from the link below and save it in the corresponding project folder.
```
Arnetminer : https://aminer.org/disambiguation
DBLP: https://github.com/yaya213/DBLP-Name-Disambiguation-Dataset
CiteSeerX: http://clgiles.ist.psu.edu/data
```

## How to run ND_NRL
Ensure the dataset path is configured correctly, then execute the following command.
```
cd embedding_model
python main.py
```

