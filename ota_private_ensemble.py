
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from privacy import binary_search_sigma
import torch
import numpy as np
from sklearn.metrics import f1_score
from utils import seed_everything

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

def add_channel_noise(signal, channel_snr):
    sigma_channel = torch.sqrt( torch.mean((signal ** 2), dim=0) / channel_snr )
    
    #    print(sigma_channel.unsqueeze(0).shape, sigma_channel.unsqueeze(0).repeat(signal.shape[0], 1).shape, signal.shape)
    
    res = signal + torch.normal(0, sigma_channel.unsqueeze(0).repeat(signal.shape[0], 1))

    return res

def add_privacy_noise(signal, epsilon, num_participating_clients, p):
    sigma = binary_search_sigma(0, 20, epsilon, 1e-6, num_participating_clients, p)

    res = signal + torch.normal(0, sigma, size=signal.shape)

    return res

def air_sum(signals, channel_snr):
    
    signal = torch.sum(torch.stack(signals, dim=0), dim=0)
    
    signal = add_channel_noise(signal, channel_snr)
    
    print(signal.shape)
    
    return signal

def client_model(beliefs, client_output, is_private, epsilon, num_participating_clients, p, A_t):

    res = torch.nn.functional.softmax(beliefs, dim=1)
    
    if client_output == "label":
        res = torch.nn.functional.one_hot(res.argmax(dim=1), beliefs.shape[1])
    
    if is_private:
        res = add_privacy_noise(res, epsilon, num_participating_clients, p)
    
    res = A_t * res
    
    return res

def server_model(signal, A_t):
    
    return (signal / A_t).argmax(dim=1)

def get_avg_score(data_name, num_repeats, num_devices, p, A_t, client_output, is_private, epsilon, channel_snr):
    results = get_results(data_name=data_name, num_devices=num_devices, num_repeats=num_repeats, main_dir="results")
    
    total_score = 0.0
    for seed_idx, (y_test_beliefs_dict, y_test_true) in get_y_test_beliefs(results, num_repeats).items():
        
        print(f"Seed {seed_idx}")
        seed_everything(seed_idx)
        participating_clients = select_participating_devices(p, num_devices)
        print("# Participating Clients: ",len(participating_clients))
        
        participating_client_beliefs = [client_model(y_test_beliefs_dict[device_idx], client_output, is_private, epsilon, len(participating_clients), p, A_t) for device_idx in participating_clients]
        
        received_signal = air_sum(participating_client_beliefs, channel_snr)
        
        y_test_pred = server_model(received_signal, A_t)
        
        total_score += calculate_score(y_test_true, y_test_pred)
    
    return total_score / num_repeats

def get_avg_score_single_model(data_name, num_repeats, num_devices, p, A_t, client_output, is_private, epsilon, channel_snr):
    results = get_results(data_name=data_name, num_devices=num_devices, num_repeats=num_repeats, main_dir="results")
    
    total_score = 0.0
    for seed_idx, (y_test_beliefs_dict, y_test_true) in get_y_test_beliefs(results, num_repeats).items():
        
        print(f"Seed {seed_idx}")
        seed_everything(seed_idx)
        
        for device_idx in range(num_devices):
            client_beliefs = client_model(y_test_beliefs_dict[device_idx], client_output, is_private, epsilon, 1, p, A_t)
        
            received_signal = add_channel_noise(client_beliefs, channel_snr) # air_sum(client_beliefs, channel_snr)
        
            y_test_pred = server_model(received_signal, A_t)
        
            total_score += calculate_score(y_test_true, y_test_pred)
    
    return total_score / (num_repeats*num_devices)

if __name__ == "__main__":
    num_repeats = 5
    num_devices = 20
    data_name = "cifar10"
    p = 0.5
    A_t = 1.0
    client_output = "belief" # belief or label
    epsilon = 1.0
    is_private = True
    channel_snr = 10.0
    
    avg_score = get_avg_score(data_name, num_repeats, num_devices, p, A_t, client_output, is_private, epsilon, channel_snr)
    
    print(avg_score)