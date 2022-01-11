from ota_private_ensemble import get_avg_score, get_avg_score_single_model
from utils import save_txt
import numpy as np

def ota_method(data, client_output, is_private):
    
    return get_avg_score(data_name=data, num_repeats=5, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=1.0, channel_snr=10)

def single_model(data, client_output, is_private):

    return get_avg_score_single_model(data_name=data, num_repeats=5, num_devices=20, p=0.5, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=1.0, channel_snr=10)

# TODO: add privacy noise/m
if __name__ == "__main__":
    items = {
        r"Single Vote ($\epsilon=\infty$)": {
            "Cifar-10": single_model("cifar10", "label", False),
            "Cifar-100": single_model("cifar100", "label", False),
            "Mnist": single_model("mnist", "label", False),
            #"Fashion-Mnist": single_model("fashionmnist", "label", False)
        },
        r"Single Vote ($\epsilon=1$)": {
            "Cifar-10": single_model("cifar10", "label", True),
            "Cifar-100": single_model("cifar100", "label", True),
            "Mnist": single_model("mnist", "label", True),
            #"Fashion-Mnist": single_model("fashionmnist", "label", True)
        },
        r"OTA Majority Voting ($\epsilon=\infty$)": {
            "Cifar-10": ota_method("cifar10", "label", False),
            "Cifar-100": ota_method("cifar100", "label", False),
            "Mnist": ota_method("mnist", "label", False),
            #"Fashion-Mnist": ota_method("fashionmnist", "label", False)
        }, 
        r"OTA Belief Summation ($\epsilon=\infty$)": {
            "Cifar-10": ota_method("cifar10", "belief", False),
            "Cifar-100": ota_method("cifar100", "belief", False),
            "Mnist": ota_method("mnist", "belief", False),
            #"Fashion-Mnist": ota_method("fashionmnist", "belief", False)
        },
        r"OTA Majority Voting ($\epsilon=1$)": {
            "Cifar-10": ota_method("cifar10", "label", True),
            "Cifar-100": ota_method("cifar100", "label", True),
            "Mnist": ota_method("mnist", "label", True),
            #"Fashion-Mnist": ota_method("fashionmnist", "label", True)
        }, 
        r"OTA Belief Summation ($\epsilon=1$)": {
            "Cifar-10": ota_method("cifar10", "belief", True),
            "Cifar-100": ota_method("cifar100", "belief", True),
            "Mnist": ota_method("mnist", "belief", True),
            #"Fashion-Mnist": ota_method("fashionmnist", "belief", True)
        },
    }


    datasets = list(items.values())[0].keys()

    res = r"""\begin{table}[htbp!]
    \centering
    \caption{Ablation study of the introduced ensemble methods}
    \begin{tabular}{""" + "l" + "c"*len(datasets) + r"""}"""

    res += r"""
    \toprule
    Method & """ + " & ".join(datasets) + r"\\" + "\n" r"\midrule \\"  + "\n"

    for method_name, item in items.items():
        res += method_name + " & "
        scores = []
        for dataset in datasets:
            max_score = np.max([curitem[dataset] for curitem in items.values()])
            if item[dataset] == max_score:
                scores.append(r"{\bf" +"{:.2f}".format(item[dataset] * 100) + "}")
            else:
                scores.append("{:.2f}".format(item[dataset]*100))
        
        res += " & ".join(scores) + r"\\" + "\n"

    res += r"""\bottomrule
    \end{tabular}
    \label{tab:ablation_study}
    \end{table}"""

    save_txt("figures", "table_ablation_study.tex", res)