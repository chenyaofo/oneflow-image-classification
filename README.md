# Oneflow Image Classification Codebase

This project aims to provide a codebase for the image classification task implemented by Oneflow.
The codebase leverages both data, tensor, and pipeline parallelism based on Oneflow to greately accelrate the training of large deep learning models.

## Requirements

Please follow `pip install -r requirements.txt`

## Get Started

You can get started with a vision transformer on imagenet (with format ofrecord) with the following commands.

**Single node, single GPU:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m entry.run --conf conf/vit.conf -o output/vit
```

> Tips: run `CUDA_VISIBLE_DEVICES=0 python -m entry.run --conf conf/vit-benchmark.conf -o output/benchmark` to check throughput performance, more details can be found at [doc/benchmark.md](doc/benchmark.md)

You can use multiple GPUs to accelerate the training with distributed data parallel:

**Single node, multiple GPUs:**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m oneflow.distributed.launch --nproc_per_node 2 \
-m entry.run --world-size 2 --conf conf/vit.conf -o output/vit
```

**Multiple nodes:**

Node 0:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m oneflow.distributed.launch --nnodes=2 \
--node_rank=0 --nproc_per_node=2 --master_addr="192.168.1.1" --master_port=7788 \
-m entry.run --conf conf/vit.conf -o output/vit
```

Node 1:
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m oneflow.distributed.launch --nnodes=2 \
--node_rank=1 --nproc_per_node=2 --master_addr="192.168.1.1" --master_port=7788 \
-m entry.run --conf conf/vit.conf -o output/vit
```


## Highlights

### Easy-to-use Config System

This codebase adopt configuration file (`.hocon`) to store the hyperparameters (such as the learning rate, training epochs and etc.).
If you want to modify the configuration hyperparameters, you have two ways:

1. Modify the configuration file to generate a new file.

2. You can add `-M` in the running command line to modify the hyperparameters temporarily.


For example, if you hope to modify the total training epochs to 100 and the learning rate to 0.05. You can run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m entry.run --conf conf/vit.conf -o output/vit -M max_epochs=100 optimizer.lr=0.05
```

If you modify a non existing hyperparameter, the code will raise an exception.

To list all valid hyperparameters names, you can run the following command:

```bash
pyhocon -i conf/vit.conf -f properties
```

### Extramely Efficient Training Powered by Oneflow

 - Auto Parallelism

 - Static Graph Optimization

 - Mixed Precision Training

 - Activation Checkpointing

 - Zero Redundancy Optimizer

 - Gradient Accumulation

Finally, enjoy the code.


## Cite

```
@misc{chen2020image,
  author = {Yaofo Chen},
  title = {Image Classification Codebase},
  year = {2023},
  howpublished = {\url{https://github.com/chenyaofo/oneflow-image-classification}}
}
```