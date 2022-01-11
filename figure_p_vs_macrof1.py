import numpy as np
from ota_private_ensemble import get_avg_score

from privacy import binary_search_sigma
from utils import save_txt

datasets = {
    "cifar10": {
        "name": "Cifar-10",
        "ymin": 0.0,
        "ymax": 1.0,
        "legendpos": r"at={(0.98,0.98)}, anchor=north east}"
    },
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
    }
}
for dataset_name, dataset in datasets.items():
    plot_begin = r"""
    \begin{figure}
    \centering
    \begin{tikzpicture}[scale=0.8]

    \definecolor{color0}{rgb}{0.12156862745098,0.466666666666667,0.705882352941177}
    \definecolor{color1}{rgb}{1,0.498039215686275,0.0549019607843137}
    \definecolor{color2}{rgb}{0.172549019607843,0.627450980392157,0.172549019607843}
    \definecolor{color3}{rgb}{0.83921568627451,0.152941176470588,0.156862745098039}
    \definecolor{color4}{rgb}{0.580392156862745,0.403921568627451,0.741176470588235}
    \definecolor{color5}{rgb}{0.580392156862745,1.0,0.741176470588235}

    \begin{axis}[
    legend cell align={left},
    legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=white!80!black, """ + dataset["legendpos"] + r""",
    log basis x={10},
    log basis y={10},
    tick align=outside,
    tick pos=left,
    x grid style={white!69.0196078431373!black},
    xmin=0.1, xmax=1.0,
    xmode=linear,
    xtick style={color=black},
    y grid style={white!69.0196078431373!black},
    ymin=""" + str(dataset["ymin"]) + r""", ymax=""" + str(dataset["ymax"])  + r""",
    ymode=linear,
    ytick style={color=black},xlabel={\small Participation Probability $p$},
    ylabel={\small Macro-F1},
    grid
    ]
    """

    plot_end = r"""
    \end{axis}
    \end{tikzpicture}
    \caption{Performance on the """ + dataset["name"] + r""" dataset for the introduced methods for varying participation probability $p$.}
    \label{fig:p_vs_macrof1}
    \end{figure}
    """

    def p_vs_macro_f1_method(data, client_output, is_private):

        
        return lambda p: get_avg_score(data_name=data, num_repeats=5, num_devices=20, p=p, A_t=1.0, client_output=client_output, is_private=is_private, epsilon=1.0, channel_snr=10)

    if __name__ == "__main__":
        items = {
            r"OTA Majority Voting ($\epsilon=\infty$)": p_vs_macro_f1_method(dataset_name, "label", False),
            r"OTA Belief Summation ($\epsilon=\infty$)": p_vs_macro_f1_method(dataset_name, "belief", False),
            r"OTA Majority Voting ($\epsilon=1$)": p_vs_macro_f1_method(dataset_name, "label", True),
            r"OTA Belief Summation ($\epsilon=1$)": p_vs_macro_f1_method(dataset_name, "belief", True),
        }
        
        n = 20
        res = "" + plot_begin
        for idx, (method, item) in enumerate(items.items()):
            res += r"""\addlegendentry{\footnotesize = """ + method + r"""}
            \addplot [very thick, color""" + str(idx) + r""", mark=x, mark size=3, mark options={solid}]
            table {
    """
            
            for p in np.arange(0.1, 1.01, 0.1):
                macro_f1 = item(p)
                res += "{:.2f}".format(p) + "\t" + "{:.3f}".format(macro_f1) + "\n"
            
            res += r"""};
            """

        res += plot_end
            
        save_txt("figures", f"figure_p_vs_macrof1_{dataset_name}.tex", res)