
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import torch
import numpy as np
from sklearn.metrics import f1_score

def get_results(data_name, num_devices, num_repeats, main_dir="results"):
    results = {}
    for seed_idx in range(num_repeats):
        results[seed_idx] = { }
        for device_idx in range(num_devices):
            
            target_path = os.path.join(main_dir, f"{data_name}_{num_devices}devices_seed{seed_idx}", f"{device_idx}.pth")

            results[seed_idx][device_idx] = torch.load(target_path, map_location=torch.device('cpu'))
    
    return results

def get_y_test_beliefs(results, num_repeats):
    res = {}
    for seed_idx in range(num_repeats):
        res[seed_idx] = ({ 
                device_idx: seed_dict["y_test_pred_beliefs"] for device_idx, seed_dict in results[seed_idx].items()
            },
            results[seed_idx][0]["y_test_true"]
        )

    return res

def calculate_score(y_true, y_pred):
    
    return f1_score(y_true, y_pred, average="macro")

def select_participating_devices(p, num_devices):

    participating_clients = []
    for device_idx in range(num_devices):
        rnd = np.random.uniform(0, 1)
        if rnd < p:
            participating_clients.append(device_idx)
    
    if len(participating_clients) == 0:
        participating_clients.append(np.random.choice(list(range(num_devices))))
    
    return participating_clients
            

if __name__ == "__main__":
    num_repeats = 10
    num_devices = 20
    data_name = "cifar10"
    
    results = get_results(data_name=data_name, num_devices=num_devices, num_repeats=num_repeats, main_dir="results")
    
    for seed_idx, (y_test_beliefs_dict, y_test_true) in get_y_test_beliefs(results, num_repeats).items():
        
        print(f"Seed {seed_idx}")
        for device_idx, y_test_beliefs in y_test_beliefs_dict.items():
            
            print(device_idx, y_test_true.shape, y_test_beliefs.shape)
            y_test_labels = torch.nn.functional.one_hot(y_test_beliefs.argmax(dim=1), y_test_beliefs.shape[1])
            print(calculate_score(y_test_true, y_test_labels.argmax(dim=1)))