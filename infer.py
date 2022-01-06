from train import datasets

for data_name in datasets.keys():
    for fname in os.listdir("results/{data_name}/"):
        if not fname.endswith(".pth"):
            continue
    
    