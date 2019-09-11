# SVD-baselines
---

### 0. About the paper

This repo is the source code of the implementations of the baselines in the paper "SVD: A Large-Scale Short Video Dataset for Near-Duplicate Video Retrieval" publised on ICCV-2019. The authors are [Qing-Yuan Jiang](http://lamda.nju.edu.cn/jiangqy), Yi He, Gen Li, Jian Lin, Lei Li and Wu-Jun Li. If you have any questions about the source code, pls contact: jiangqy#lamda.nju.edu.cn or qyjiang#gmail.com

### 1. Running Environment
```bash
python 3
pytorch
tensorflow
```

### 2. Running Demos
#### 2.0. Preliminary
##### Directory to store some files
```
path/to/data
│
└───videos/
│    │ xxx.mp4
│    │ ...
│
└───frames/
│    │ xxx.lmdb
│    │ ...
│
└───ffeatures/
│    │ xxx.lmdb
│    │ ...
│
└───vfeatures/
│    │ xxx.lmdb
│    │ ...
```

#### 2.1. Preprocessing
##### Frame Extraction
##### Deep Features Extraction
##### Video Features Aggregations
#### 2.2. Real-Value based Method
##### DML
##### CNNL
##### CNNV
#### 2.3. Hashing based Method
##### LSH
##### ITQ
##### IsoH

