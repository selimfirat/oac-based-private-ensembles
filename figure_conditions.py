import numpy as np
from ota_private_ensemble import get_avg_score

from privacy import binary_search_sigma
from utils import save_txt, seed_everything
def p_vs_macro_f1_method(data, client_output, is_private):
    
    return lambda p: get_avg_score(data_name=data, num_repeats=5, num_devices=20, p=p, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=1.0, channel_snr=10)

def snr_vs_macro_f1_method(data, client_output, is_private):
    
    return lambda snr: get_avg_score(data_name=data, num_repeats=5, num_devices=20, p=1.0, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=1.0, channel_snr=snr)


datasets = {
    "cifar10": {
        "name": "Cifar-10",
        "yrange_p": (50, 100),
        "yrange_snr": (50, 100),
        "xrange_p": (0.1, 1.1, 0.1),
        "xrange_snr": (-2, 21, 1)
    },
}
"""
"cifar100": {
    "name": "Cifar-100",
    "ymin": 0.0,
    "ymax": 1.0,
    "legendpos": r"at={(0.98,0.98)}, anchor=north east}"
},
"mnist": {
    "name": "Mnist",
    "ymin": 0.0,
    "ymax": 1.0,
    "legendpos": r"at={(0.98,0.98)}, anchor=north east}"
},
"fashionmnist": {
    "name": "FashionMnist",
    "ymin": 0.0,
    "ymax": 1.0,
    "legendpos": r"at={(0.98,0.98)}, anchor=north east}"
}
"""

if __name__ == "__main__":
    for dataset_name, dataset in datasets.items():
        plot_begin = r"""
    \pgfplotsset{footnotesize,samples=10}    \definecolor{color0}{rgb}{0.12156862745098,0.466666666666667,0.705882352941177}
    \definecolor{color1}{rgb}{1,0.498039215686275,0.0549019607843137}
    \definecolor{color2}{rgb}{0.172549019607843,0.627450980392157,0.172549019607843}
    \definecolor{color3}{rgb}{0.83921568627451,0.152941176470588,0.156862745098039}
    \definecolor{color4}{rgb}{0.580392156862745,0.403921568627451,0.741176470588235}
    \definecolor{color5}{rgb}{0.580392156862745,1.0,0.741176470588235}

    \begin{figure}[htbp!]
    \centering
    \ref*{named}
    \begin{tikzpicture}[scale=0.92]
    \begin{axis}[
    legend columns=2,
    legend entries={\footnotesize Majority Voting $(\epsilon=\infty)$,\footnotesize Belief Summation $(\epsilon=\infty)$,\footnotesize Majority Voting $(\epsilon=1)$,\footnotesize Belief Summation $(\epsilon=1)$},
    legend to name=named,
    log basis x={10},
    log basis y={10},
    tick align=inside,
    tick pos=left,
    x grid style={white!69.0196078431373!black},
    xmin=-2, xmax=20,
    xmode=linear,
    xtick style={color=black},
    y grid style={white!69.0196078431373!black},
    ymin=0.75, ymax=1.0,
    ymode=linear,
    ytick style={color=black},xlabel={\footnotesize Channel SNR (dB)},
    ylabel={\footnotesize Macro-F1},
    grid
    ]
        """

        plot_mid = r"""
        \end{axis}
    \end{tikzpicture}
    %
    \begin{tikzpicture}[scale=0.9]
    \begin{axis}[
    log basis x={10},
    log basis y={10},
    tick align=inside,
    tick pos=left,
    x grid style={white!69.0196078431373!black},
    xmin=0.1, xmax=1.0,
    xmode=linear,
    xtick style={color=black},
    y grid style={white!69.0196078431373!black},
    ymin=0.35, ymax=1.0,
    ymode=linear,
    ytick style={color=black},xlabel={\footnotesize Participation Probabilty $p$},
    ylabel={\footnotesize Macro-F1},
    grid
    ]
        """

        plot_end = r"""
        \end{axis}
    \end{tikzpicture}
    \caption{Analysis of ensemble methods' performance with OOAC for varying conditions: channel SNR (left) and participation probability $p$ (right).}
    \label{fig:conditions}
    \end{figure}
        """

    snr_items = {
        r"OTA Majority Voting ($\epsilon=\infty$)": snr_vs_macro_f1_method(dataset_name, "label", False),
        r"OTA Belief Summation ($\epsilon=\infty$)": snr_vs_macro_f1_method(dataset_name, "belief", False),
        r"OTA Majority Voting ($\epsilon=1$)": snr_vs_macro_f1_method(dataset_name, "label", True),
        r"OTA Belief Summation ($\epsilon=1$)": snr_vs_macro_f1_method(dataset_name, "belief", True),
    }

    n = 20
    res = "" + plot_begin
    for idx, (method, item) in enumerate(snr_items.items()):
        res += r"""
        \addplot [""" + ("densely dashed" if idx % 2 else "densely dotted") + r""", semithick, color""" + str(idx) + r""", mark=""" + ("o" if idx % 2 else "x") +  r""", mark size=1, mark options={solid}]
        table {
"""
        
        for snr_db in np.arange(-2, 21, 2):
            seed_everything(1)
            macro_f1 = item(10**(0.1*snr_db))[0] # convert db to original
            res += str(snr_db) + "\t" + "{:.4f}".format(macro_f1) + "\n"
        
        res += r"""};
        """
    
    res += plot_mid
    participation_items = {
        r"Majority Voting ($\epsilon=\infty$)": p_vs_macro_f1_method(dataset_name, "label", False),
        r"Belief Summation ($\epsilon=\infty$)": p_vs_macro_f1_method(dataset_name, "belief", False),
        r"Majority Voting ($\epsilon=1$)": p_vs_macro_f1_method(dataset_name, "label", True),
        r"Belief Summation ($\epsilon=1$)": p_vs_macro_f1_method(dataset_name, "belief", True),
    }
    
    for idx, (method, item) in enumerate(participation_items.items()):
        res += r"""
        \addplot [""" + ("densely dashed" if idx % 2 else "densely dotted") + r""",thick, color""" + str(idx) + r""", mark=""" + ("o" if idx % 2 else "x") +  r""", mark size=1, mark options={solid}]
        table {
"""
        for p in np.arange(0.1, 1.01, 0.1):
            seed_everything(1)
            macro_f1 = item(p)[0]
            res += "{:.2f}".format(p) + "\t" + "{:.4f}".format(macro_f1) + "\n"
        
        res += r"""};
        """

    res += plot_end
        
    save_txt("figures", f"figure_conditions_{dataset_name}.tex", res)