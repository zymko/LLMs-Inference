import torch
from torch.utils.data import Dataset, DataLoader
from log.log import create_logger
from data.dataset import Matrix
import yaml
import torch.nn.functional as F
import time
from monitor import GPUUsageMonitor

def attention(queries, keys, values):
    d_k = queries.size(-1)
    keys_T = keys.transpose(1, 2)
    return F.softmax(queries @ keys_T / (d_k ** 0.5) , dim=1) @ values

def inference(dataloader, device):
    for query, key, value in dataloader:
        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        attention(queries=query, keys=key, values=value)
        torch.cuda.empty_cache()
        


if __name__ == '__main__':


    logger = create_logger(name='__name__', log_file='src/log/run.log')
    logger.info('Logger created successfully!')
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    d_v = config['matrix']['d_v']
    d_k = config['matrix']['d_k']
    tokens = config['matrix']['tokens']
    dataset_size = config['dataset']['size']
    logger.info(f'Dimension of queries is {d_k}, dimensions of values is {d_v}, number of tokens is {tokens}, datset size is {dataset_size}')
    dataset = Matrix(matrix_sizes=((tokens, d_k), (tokens, d_k), (tokens, d_v)), size=dataset_size, seeds=(111, 222, 333))
    logger.info('Queries, keys and values initialized successfully')
    dataloader = DataLoader(dataset=dataset, batch_size=512, num_workers=16, shuffle=False)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    

    logger.info('Starting:')

    # Reset the peak memory stats
    torch.cuda.reset_peak_memory_stats()

    # # starting the monitoring
    start_time = time.time()
    monitor = GPUUsageMonitor(interval=1)

    monitor.start()

    inference(dataloader, device)

    end_time = time.time()
    monitor.stop()
    avg_utilization = monitor.average_utilization()
    print(f"Average GPU Utilization: {avg_utilization:.2f}%")
    print(f'GPU utilization rate: {monitor.utilization_data}')

    elapsed_time = end_time - start_time
    logger.info(f'Running time: {elapsed_time}')

    # Print peak GPU memory usage
    print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    # Optionally clear cache
    torch.cuda.empty_cache()
        

    # embeddings = attention(queries, values, keys)


