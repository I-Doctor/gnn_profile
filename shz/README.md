# Usage

Download the source codes of [DGL 0.7.x](https://github.com/dmlc/dgl/tree/0.7.x) and copy *run_time.py* and *run_profile.py* into the directory as follows,
```
(path to dgl)/examples/pytorch/model_zoo/citation_network/
```
To see the running times in diferent stages, run this,
```
python run_time.py --gpu 0 --model chebnet --dataset cora --self-loop
```
To see the profiling results of GCN, run this, 
```
python run_profile.py --gpu 0 --model chebnet --dataset cora --self-loop
```

The model can also be sgc, gin, or any other models in the model_zoo. And the dataset can be cora, citeseer, pubmed. The '--self-loop' is optional. 
