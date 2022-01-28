from ota_private_ensemble import get_avg_score, get_avg_score_different_channels, get_avg_score_single_model
from utils import save_txt
import numpy as np
from utils import seed_everything

num_repeats = 5

def ota_method(data, client_output, is_private, epsilon=1.0, apply_softmax=False):
    seed_everything(1)
    
    return get_avg_score(data_name=data, num_repeats=num_repeats, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=epsilon, channel_snr=10, apply_softmax=apply_softmax)

def single_model(data, client_output, is_private, epsilon=1.0, apply_softmax=False):
    seed_everything(1)

    return get_avg_score_single_model(data_name=data, num_repeats=num_repeats, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=epsilon, channel_snr=10, apply_softmax=apply_softmax)

def orthogonal_model(data, client_output, is_private, epsilon=1.0, apply_softmax=False):
    seed_everything(1)

    return get_avg_score_different_channels(data_name=data, num_repeats=num_repeats, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=epsilon, channel_snr=10, apply_softmax=apply_softmax)

def generate_dict(shown_datasets, method, *attrs):
    res = {}
    for name, key in shown_datasets.items():
        res[name] = method(key, *attrs, apply_softmax=key in ["yelp_polarity", "emotion", "yelp_review_full", "imdb"])

    return res

def print_items(items):
    datasets = list(items.values())[0].keys()

    res = ""
    for method_name, item in items.items():
        res += method_name + " & "
        scores = []
        for dataset in datasets:
            max_score = np.max([curitem[dataset][0] for curitem in items.values()])
            if np.around(item[dataset][0], 4) == np.around(max_score, 4):
                scores.append(r"$\mathbf{" + "{:.2f}".format(item[dataset][0] * 100) + r" {\scriptscriptstyle \pm " + "{:.2f}".format(item[dataset][1] * 100) + "}}$")
            else:
                scores.append("${:.2f}".format(item[dataset][0]*100)  + r" {\scriptscriptstyle \pm " + "{:.2f}".format(item[dataset][1] * 100) + "}$")
        
        res += " & ".join(scores) + r"\\" + "\n"

    return res

if __name__ == "__main__":
    seed_everything(1)
    shown_datasets = {
        "Cifar10": "cifar10",
        "Cifar100": "cifar100",
        #"Mnist": "mnist",
        "FashionMnist": "fashionmnist",
        #"Yelp": "yelp_polarity",
        #"Emotion": "emotion",
        "Imdb": "imdb",
    }

    items = {
        r"Single Best Vote ($\epsilon=\infty$)": generate_dict(shown_datasets, single_model, "label", False),
        r"Orthogonal Majority Voting ($\epsilon=\infty$)": generate_dict(shown_datasets, orthogonal_model, "label", False),
        r"Orthogonal Belief Summation ($\epsilon=\infty$)": generate_dict(shown_datasets, orthogonal_model, "belief", False),
        r"Majority Voting with OAC ($\epsilon=\infty$)": generate_dict(shown_datasets, ota_method, "label", False), 
        r"Belief Summation with OAC ($\epsilon=\infty$)":  generate_dict(shown_datasets, ota_method, "belief", False),
    }

    items_private = {
        r"Single Best Vote ($\epsilon=1$)": generate_dict(shown_datasets, single_model, "label", True),
        r"Orthogonal Majority Voting ($\epsilon=1$)": generate_dict(shown_datasets, orthogonal_model, "label", True),
        r"Orthogonal Belief Summation ($\epsilon=1$)": generate_dict(shown_datasets, orthogonal_model, "belief", True),
        r"Majority Voting with OAC ($\epsilon=1$)": generate_dict(shown_datasets, ota_method, "label", True), 
        r"Belief Summation with OAC ($\epsilon=1$)":  generate_dict(shown_datasets, ota_method, "belief", True),
    }

    items_weakprivate = {
        r"Single Best Vote ($\epsilon=10$)": generate_dict(shown_datasets, single_model, "label", True, 10),
        r"Orthogonal Majority Voting ($\epsilon=10$)": generate_dict(shown_datasets, orthogonal_model, "label", True, 10),
        r"Orthogonal Belief Summation ($\epsilon=10$)": generate_dict(shown_datasets, orthogonal_model, "belief", True, 10),
        r"Majority Voting with OAC ($\epsilon=10$)": generate_dict(shown_datasets, ota_method, "label", True, 10), 
        r"Belief Summation with OAC ($\epsilon=10$)":  generate_dict(shown_datasets, ota_method, "belief", True, 10),
    }

    datasets = list(items.values())[0].keys()

    res = r"""\begin{table}[htbp!]
    \centering
    \caption{Comparison with the Baselines}
    \resizebox{0.45\textwidth}{!}{\begin{tabular}{""" + "l" + "c"*len(datasets) + r"""}"""

    res += r"""
    \toprule
    Method & """ + " & ".join(datasets) + r"\\ \midrule"  + "\n"
    
    res += print_items(items)

    res += r"""\midrule""" + "\n"
    
    res += print_items(items_weakprivate)

    res += r"""\midrule""" + "\n"
    
    res += print_items(items_private)

    res += r"""\bottomrule
    \end{tabular}}
    \label{tab:comparison}
    \end{table}"""

    save_txt("figures", "table_comparison.tex", res)