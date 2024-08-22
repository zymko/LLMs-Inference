# LLMs-Inference

## Dataset
The tree structure of dataset is as follows:
```bash
dataset
├── 1200-400
│   ├── key
│   ├── query
│   └── value
├── 1600-400
│   ├── key
│   ├── query
│   └── value
├── 2000-400
│   ├── key
│   ├── query
│   └── value
├── 400-400
│   ├── key
│   ├── query
│   └── value
└── 800-400
    ├── key
    ├── query
    └── value
```
The dataset contains five folders, each of which stores 3*2000 matrices of a given size indicating by folder's name. Consider the length of tokens is usually larger than the length of embeddings, we generated these matrix with rows larger than columns. The dataset is too large to upload to github, but one can use utils/data.py to generate this dataset, in which values are stored as Float32. Please put dataset under the root path before running a test.

## Matrix Multiplication
Currently, our test is based on a [Scaled Dot-Product Attention](https://example.com)
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

## Results

### Memory

#### Statistics
|  Size                  | 8    | 16   | 32   | 64   | 128  | 256  | 512  |
|------------------------|------|------|------|------|------|------|------|
| 400x400                | 12.85| 21.00| 31.43| 41.97| 52.43| 62.46| 72.39|
| 800x400                | 13.66| 21.09| 31.39| 41.97| 52.46| 63.41| 73.25|
| 1200x400               | 12.15| 21.06| 31.42| 42.14| 52.66| 63.53| 74.44|
| 1600x400               | 11.38| 21.31| 31.55| 42.11| 52.45| 63.35| 74.18|
| 2000x400               | 10.54| 21.59| 32.08| 43.28| 53.64| 64.77| 75.92|
| 800x400                | 10.82| 22.32| 33.48| 45.92| 57.77| 69.54| 81.31|
| 1600x400               | 16.14| 24.27| 37.13| 49.20| 61.75| 74.26| 86.77|


#### Fix Matrix Size
We fixed the size of queries, keys and values, while changing batch size from 8 to 512. The following image shows the relationship between memory in GB with batch size, given a fixed size of queries, keys and values
![Memory & Matrix Size](results/images/memory_matrix_size.png)

#### Fix Batch Size
We fixed the batch size, while changing matrix (query, value, key) size from 400x400 to 2000x400. The following image shows the relationship between memory in GB with matrix size, given a fixed batch size
![Memory & Batch Size](results/images/memory_batch_size.png)


### Running Time

#### Statistics

#### Fix Matrix Size
We fixed the size of queries, keys and values, while changing batch size from 8 to 512. The following image shows the relationship between running time in second with batch size, given a fixed size of queries, keys and values
![Running time & Matrix Size](results/images/time_matrix_size.png)

#### Fix Batch Size
We fixed the batch size, while changing matrix (query, value, key) size from 400x400 to 2000x400. The following image shows the relationship between running time in second with matrix size, given a fixed batch size
![Running time & Batch Size](results/images/time_batch_size.png)




## Conlcusion