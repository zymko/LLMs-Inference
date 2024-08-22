# LLMs-Inference

## Dataset

## 

## Results

### Memory

#### Fix Matrix Size
We fixed the size of queries, keys and values, while changing batch size from 8 to 512. The following image shows the relationship between memory in GB with batch size, given a fixed size of queries, keys and values
![Memory & Batch Size](results/images/memory_matrix_size.png)

#### Fix Batch Size
We fixed the batch size, while changing matrix (query, value, key) size from 400x400 to 2000x400. The following image shows the relationship between memory in GB with matrix size, given a fixed batch size
![Memory & Batch Size](results/images/memory_batch_size.png)