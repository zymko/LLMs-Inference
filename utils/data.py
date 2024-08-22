import numpy as np
import yaml
import os
import gc

if __name__ == '__main__':

    with open('data_generate.yaml', 'r') as file:
        config = yaml.safe_load(file)

    size = config['dataset']['size']
    d_v = config['matrix']['d_v']
    d_k = config['matrix']['d_k']
    tokens = config['matrix']['tokens']

    for token in tokens:

        matrix_shape_query = (size, token, d_k)
        matrix_shape_key = (size, token, d_k)
        matrix_shape_value = (size, token, d_v)

        folder = f'dataset/{token}-{d_k}'
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f'{folder}/query', exist_ok=True)
        os.makedirs(f'{folder}/key', exist_ok=True)
        os.makedirs(f'{folder}/value', exist_ok=True)

        def save_matrix_slices(matrix, folder_name, prefix):
            for i in range(matrix.shape[0]):
                filename = os.path.join(folder_name, f"{prefix}_slice_{i}.npy")
                np.save(filename, matrix[i])
                print(f"Saved {filename}")

        # Generate and save query matrices
        np.random.seed(111)
        query_matrix = np.random.rand(*matrix_shape_query).astype(np.float32)
        save_matrix_slices(query_matrix, f'{folder}/query', 'query')
        del query_matrix
        gc.collect()  

        # Generate and save key matrices
        np.random.seed(222)
        key_matrix = np.random.rand(*matrix_shape_key).astype(np.float32)
        save_matrix_slices(key_matrix, f'{folder}/key', 'key')
        del key_matrix
        gc.collect()  

        # Generate and save value matrices
        np.random.seed(333)
        value_matrix = np.random.rand(*matrix_shape_value).astype(np.float32)
        save_matrix_slices(value_matrix, f'{folder}/value', 'value')
        del value_matrix
        gc.collect()  