import torch
from torch.utils.data import Dataset, DataLoader
from utils.log import create_logger
from utils.dataset import Matrix, GeneratedMatrix
import yaml
import torch.nn.functional as F
import time
from utils.monitor import GPUUsageMonitor
import csv
import tqdm

def attention(queries, keys, values):
    d_k = queries.size(-1)
    keys_T = keys.transpose(1, 2)
    return F.softmax(queries @ keys_T / (d_k ** 0.5) , dim=1) @ values

def inference(dataloader, device):
    for query, key, value in tqdm.tqdm(dataloader):
        query = query.to(device)
        key = key.to(device)
        value = value.to(device)
        attention(queries=query, keys=key, values=value)

def batch_test_inference(batch_size, config, device, folder='2000-400'):  
    dataset = GeneratedMatrix(query_dir=f'dataset/{folder}/query', key_dir=f'dataset/{folder}/key', value_dir=f'dataset/{folder}/value')
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    inference(dataloader, device)

def matrix_test_inference(folder, config, device, batch_size=32):  
    dataset = GeneratedMatrix(query_dir=f'dataset/{folder}/query', key_dir=f'dataset/{folder}/key', value_dir=f'dataset/{folder}/value')
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    inference(dataloader, device)

def batch_test():
    with open('batch.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    batch_sizes = config['dataset']['batch_sizes']
    running_time = []
    memory =[]
    for batch_size in batch_sizes:

        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # starting the monitoring
        start_time = time.time()
        batch_test_inference(batch_size, config, device, folder='2000-400')
        end_time = time.time()

        elapsed_time = end_time - start_time
        # elapsed_time = sum(times)
        logger.info(f'Running time: {elapsed_time}')
        

        # Print peak GPU memory usage
        logger.info(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        running_time.append(elapsed_time)
        memory.append(torch.cuda.max_memory_allocated()/ 1024 ** 2)
        # Optionally clear cache
        torch.cuda.empty_cache()
        
    # with open('results/batch_size/memory.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(memory)
    # with open('results/batch_size/time.csv', 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(running_time)

def matrix_test():
    with open('matrix.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    tokens = config['matrix']['tokens']

    running_time = []
    memory =[]
    batch_size = 8
    for token in tokens:

        # Reset the peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # starting the monitoring
        start_time = time.time()
        matrix_test_inference(f'{token}-400', config, device, batch_size=batch_size)
        end_time = time.time()

        elapsed_time = end_time - start_time
        logger.info(f'Running time: {elapsed_time}') 

        # Print peak GPU memory usage
        logger.info(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
        running_time.append(elapsed_time)
        memory.append(torch.cuda.max_memory_allocated()/ 1024 ** 2)
        # Optionally clear cache
        torch.cuda.empty_cache()
        
    with open(f'results/matrix_size/memory_batch_{batch_size}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(memory)
    with open(f'results/matrix_size/time_batch_{batch_size}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(running_time)

if __name__ == '__main__':

    logger = create_logger(name='__name__', log_file='log.log')
    logger.info('Logger created successfully!')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    logger.info('Starting:')

    matrix_test()
    # batch_test()

    
    


   
  
    

    

        


