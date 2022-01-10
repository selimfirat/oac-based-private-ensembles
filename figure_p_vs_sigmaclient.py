import numpy as np

from privacy import binary_search_sigma
from utils import save_txt

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
legend style={fill opacity=0.8, draw opacity=1, text opacity=1, draw=white!80!black, at={(0.98,0.98)}, anchor=north east},
log basis x={10},
log basis y={10},
tick align=outside,
tick pos=left,
x grid style={white!69.0196078431373!black},
xmin=0.1, xmax=1.0,
xmode=linear,
xtick style={color=black},
y grid style={white!69.0196078431373!black},
ymin=0, ymax=3,
ymode=linear,
ytick style={color=black},xlabel={\small Participation Probability $p$},
ylabel={\small Required $\sigma_{\mathrm{client}}$},
grid
]
"""

plot_end = r"""
\end{axis}
\end{tikzpicture}
\caption{Participation probability $p$ vs required $\sigma_{\mathmrm{client}}$ for various $\epsilon$.}
\label{fig:p}
\end{figure}
"""

if __name__ == "__main__":
    n = 20
    res = "" + plot_begin
    items = [
        { "epsilon": 1, "color": "color0" },
        { "epsilon": 5, "color": "color1" },
        { "epsilon": 10, "color": "color2" },
        { "epsilon": 100, "color": "color3" },
    ]
    for item in items:
        res += r"""\addlegendentry{\footnotesize $\epsilon = """ + str(item["epsilon"]) + r""", \delta=10^{-6}$}
        \addplot [very thick, """ + item["color"] + r""", mark=x, mark size=3, mark options={solid}]
        table {%"""
        
        for p in np.arange(0.1, 1.01, 0.1):
            sigma = binary_search_sigma(0, 20, item["epsilon"], 1e-6, p*n, p)
            res += "{:.2f}".format(p) + "\t" + "{:.3f}".format(sigma) + "\n"
        
        res += r"""};
        """

    res += plot_end
        
    save_txt("figures", "figure_p_vs_sigmaclient.tex", res)