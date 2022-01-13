from ota_private_ensemble import get_avg_score, get_avg_score_different_channels, get_avg_score_single_model
from utils import save_txt
import numpy as np

def ota_method(data, client_output, is_private, epsilon=1.0):
    
    return get_avg_score(data_name=data, num_repeats=5, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=epsilon, channel_snr=10)

def single_model(data, client_output, is_private, epsilon=1.0):

    return get_avg_score_single_model(data_name=data, num_repeats=5, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=epsilon, channel_snr=10)

def orthogonal_model(data, client_output, is_private, epsilon=1.0):

    return get_avg_score_different_channels(data_name=data, num_repeats=5, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=epsilon, channel_snr=10)

if __name__ == "__main__":
    items = {
        r"Single Best Vote ($\epsilon=\infty$)": {
            "Cifar-10": single_model("cifar10", "label", False),
            "Mnist": single_model("mnist", "label", False),
            "Fashion-Mnist": single_model("fashionmnist", "label", False)
        },
        r"Orthogonal Majority Voting ($\epsilon=\infty$)": {
            "Cifar-10": orthogonal_model("cifar10", "label", False),
            "Mnist": orthogonal_model("mnist", "label", False),
            "Fashion-Mnist": orthogonal_model("fashionmnist", "label", False)
        }, 
        r"Orthogonal Belief Summation ($\epsilon=\infty$)": {
            "Cifar-10": orthogonal_model("cifar10", "belief", False),
            "Mnist": orthogonal_model("mnist", "belief", False),
            "Fashion-Mnist": orthogonal_model("fashionmnist", "belief", False)
        },
        r"OAC Majority Voting ($\epsilon=\infty$)": {
            "Cifar-10": ota_method("cifar10", "label", False),
            "Mnist": ota_method("mnist", "label", False),
            "Fashion-Mnist": ota_method("fashionmnist", "label", False)
        }, 
        r"OAC Belief Summation ($\epsilon=\infty$)": {
            "Cifar-10": ota_method("cifar10", "belief", False),
            "Mnist": ota_method("mnist", "belief", False),
            "Fashion-Mnist": ota_method("fashionmnist", "belief", False)
        },
    }

    items_private = {
        r"Single Best Vote ($\epsilon=1$)": {
            "Cifar-10": single_model("cifar10", "label", True),
            "Cifar-100": single_model("cifar100", "label", True),
            "Mnist": single_model("mnist", "label", True),
            "Fashion-Mnist": single_model("fashionmnist", "label", True)
        },
        r"Orthogonal Majority Voting ($\epsilon=1$)": {
            "Cifar-10": orthogonal_model("cifar10", "label", True),
            "Cifar-100": orthogonal_model("cifar100", "label", True),
            "Mnist": orthogonal_model("mnist", "label", True),
            "Fashion-Mnist": orthogonal_model("fashionmnist", "label", True)
        }, 
        r"Orthogonal Belief Summation ($\epsilon=1$)": {
            "Cifar-10": orthogonal_model("cifar10", "belief", True),
            "Cifar-100": orthogonal_model("cifar100", "belief", True),
            "Mnist": orthogonal_model("mnist", "belief", True),
            "Fashion-Mnist": orthogonal_model("fashionmnist", "belief", True)
        },
        r"OAC Majority Voting ($\epsilon=1$)": {
            "Cifar-10": ota_method("cifar10", "label", True),
            "Cifar-100": ota_method("cifar100", "label", True),
            "Mnist": ota_method("mnist", "label", True),
            "Fashion-Mnist": ota_method("fashionmnist", "label", True)
        }, 
        r"OAC Belief Summation ($\epsilon=1$)": {
            "Cifar-10": ota_method("cifar10", "belief", True),
            "Cifar-100": ota_method("cifar100", "belief", True),
            "Mnist": ota_method("mnist", "belief", True),
            "Fashion-Mnist": ota_method("fashionmnist", "belief", True)
        },
    }

    items_weakprivate = {
        r"Single Best Vote ($\epsilon=10$)": {
            "Cifar-10": single_model("cifar10", "label", True, 10),
            "Cifar-100": single_model("cifar100", "label", True, 10),
            "Mnist": single_model("mnist", "label", True, 10),
            "Fashion-Mnist": single_model("fashionmnist", "label", True, 10)
        },
        r"Orthogonal Majority Voting ($\epsilon=10$)": {
            "Cifar-10": orthogonal_model("cifar10", "label", True, 10),
            "Cifar-100": orthogonal_model("cifar100", "label", True, 10),
            "Mnist": orthogonal_model("mnist", "label", True, 10),
            "Fashion-Mnist": orthogonal_model("fashionmnist", "label", True, 10)
        }, 
        r"Orthogonal Belief Summation ($\epsilon=10$)": {
            "Cifar-10": orthogonal_model("cifar10", "belief", True, 10),
            "Cifar-100": orthogonal_model("cifar100", "belief", True, 10),
            "Mnist": orthogonal_model("mnist", "belief", True, 10),
            "Fashion-Mnist": orthogonal_model("fashionmnist", "belief", True, 10)
        },
        r"OAC Majority Voting ($\epsilon=10$)": {
            "Cifar-10": ota_method("cifar10", "label", True, 10),
            "Cifar-100": ota_method("cifar100", "label", True, 10),
            "Mnist": ota_method("mnist", "label", True, 10),
            "Fashion-Mnist": ota_method("fashionmnist", "label", True, 10)
        }, 
        r"OAC Belief Summation ($\epsilon=10$)": {
            "Cifar-10": ota_method("cifar10", "belief", True, 10),
            "Cifar-100": ota_method("cifar100", "belief", True, 10),
            "Mnist": ota_method("mnist", "belief", True, 10),
            "Fashion-Mnist": ota_method("fashionmnist", "belief", True, 10)
        },
    }

    datasets = list(items.values())[0].keys()

    res = r"""\begin{table}[htbp!]
    \centering
    \caption{Ablation study of the introduced ensemble methods}
    \resizebox{0.45\textwidth}{!}{\begin{tabular}{""" + "l" + "c"*len(datasets) + r"""}"""

    res += r"""
    \toprule
    Method & """ + " & ".join(datasets) + r"\\ \midrule"  + "\n"
    
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

    res += r"""\midrule""" + "\n"
    
    for method_name, item in items_weakprivate.items():
        res += method_name + " & "
        scores = []
        for dataset in datasets:
            max_score = np.max([curitem[dataset] for curitem in items_weakprivate.values()])
            if item[dataset] == max_score:
                scores.append(r"{\bf" +"{:.2f}".format(item[dataset] * 100) + "}")
            else:
                scores.append("{:.2f}".format(item[dataset]*100))
        
        res += " & ".join(scores) + r"\\" + "\n"

    res += r"""\midrule""" + "\n"
    
    for method_name, item in items_private.items():
        res += method_name + " & "
        scores = []
        for dataset in datasets:
            max_score = np.max([curitem[dataset] for curitem in items_private.values()])
            if item[dataset] == max_score:
                scores.append(r"{\bf" +"{:.2f}".format(item[dataset] * 100) + "}")
            else:
                scores.append("{:.2f}".format(item[dataset]*100))
        
        res += " & ".join(scores) + r"\\" + "\n"

    res += r"""\bottomrule
    \end{tabular}}
    \label{tab:ablation_study}
    \end{table}"""

    save_txt("figures", "table_ablation_study.tex", res)