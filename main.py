import torch
from torch.utils.data import Dataset, DataLoader
from utilis.log import create_logger
from utilis.dataset import Matrix, GeneratedMatrix
import yaml
import torch.nn.functional as F
import time
from utilis.monitor import GPUUsageMonitor
import csv

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


def batch_test(batch_size, config, device, folder='2000-400'):
    # d_v = config['matrix']['d_v']
    # d_k = config['matrix']['d_k']
    # tokens = config['matrix']['tokens']
    # dataset_size = config['dataset']['size']
    # logger.info(f'Dimension of queries is {d_k}, dimensions of values is {d_v}, number of tokens is {tokens}, datset size is {dataset_size}')
    # dataset = Matrix(matrix_sizes=((tokens, d_k), (tokens, d_k), (tokens, d_v)), size=dataset_size, seeds=(111, 222, 333))
    # # logger.info('Queries, keys and values initialized successfully')
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, shuffle=False)
   
    dataset = GeneratedMatrix(query_dir=f'dataset/{folder}/query', key_dir=f'dataset/{folder}/key', value_dir=f'dataset/{folder}/value')
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    inference(dataloader, device)
    
        
if __name__ == '__main__':

    logger = create_logger(name='__name__', log_file='log.log')
    logger.info('Logger created successfully!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('Starting:')

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    batch_sizes = config['dataset']['batch_sizes']

    running_time = []
    memory =[]
    

    for batch_size in batch_sizes:

        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # # starting the monitoring
        start_time = time.time()

        batch_test(batch_size, config, device)

        end_time = time.time()

        elapsed_time = end_time - start_time
        logger.info(f'Running time: {elapsed_time}')

        # Print peak GPU memory usage
        logger.info(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        running_time.append(elapsed_time)
        memory.append(torch.cuda.max_memory_allocated()/ 1024 ** 2)
        # Optionally clear cache
        torch.cuda.empty_cache()
        
    with open('results/batch_size/memory.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(memory)
    with open('results/batch_size/time.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(running_time)


   
  
    

    

        


