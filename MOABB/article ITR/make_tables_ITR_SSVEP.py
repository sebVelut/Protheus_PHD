import pandas as pd
import numpy as np
import re
from pathlib import Path


def mean_std(x):
    x = x
    mean = np.round(np.mean(x), 2)
    std = np.round(np.std(x), 2)
    return f"{mean:.2f}$\pm${std:.2f}"


def create_latex_table(df, name, base_path=None):
    if base_path is not None:
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True, parents=True)
        buf_name = base_path / f"dataset-{name}.tex"
    else:
        buf_name = f"dataset-{name}.tex"

    label = f'tab:{name}_agg_dataset'
    df_results = df.groupby(["dataset", "pipeline"])["ITR"].apply(
        mean_std).unstack()
    df_numeric = df.groupby(["dataset", "pipeline"])["ITR"].apply(
        np.mean).unstack()

    df_mean_mean = pd.Series(df_numeric.mean(axis=0), name='Average')
    df_numeric = df_numeric.append(df_mean_mean)
    avg_line = np.round(df_numeric.loc['Average'], 2)

    df_mean_dataset = pd.Series(df_numeric.mean(axis=1),name='Average')
    df_numeric = pd.concat([df_numeric, df_mean_dataset], axis=1)

    avg_col = np.round(df_numeric['Average'], 2)

    df_results = df_results.append(avg_line)
    df_results = pd.concat([df_results, avg_col], axis=1)

    df_results = df_results.T
    df_mean = df_numeric.T

    max_values = np.argmax(df_mean.to_numpy(), 0)
    for ind, (dataset, line) in enumerate(df_results.items()):
        try:
            # import pdb;pdb.set_trace()
            if dataset.lower() == "average":
                line.iloc[max_values[ind]] = '\textbf{' + str(
                    line.iloc[max_values[ind]]) + '}'
            else:
                line.iloc[max_values[ind]] = re.sub(r'(\d+\.\d+)',
                                                    r'\\textbf{\1}',
                                                    line.iloc[max_values[ind]])

        except:
            pass
    df_results = df_results.reset_index()

    df_results['index'] = df_results['index'].str.replace(" ", "").str.replace(
        "=", "").str.replace("_", "").str.replace("SSVEP", "")
    df_results.columns =  ["pipeline"] + df_results.columns[1:].tolist()
    # df_results.columns = [f"${col}$" for col in df_results.columns]
    table_config = {
        'column_format': 'c' * len(df_results.columns),
        # Specify column alignment, e.g., 'c' for center-aligned
        'escape': False,  # Disable escaping special characters
        'index': False,  # Do not include the index column
        'label': label,  # Label for referencing the table
        'caption': f'{name.upper()}',  # Caption for the table
        'longtable': False,  # Use longtable environment for multi-page tables

    }

    latex_table = df_results.to_latex(**table_config)

    latex_table = latex_table.replace("begin{table}", "begin{table*}").replace(
        "end{table}", "end{table*}")
    latex_table = latex_table.replace("label{" + label + "}",
                                      "label{" + label + "}" + "\n\\begin{adjustbox}{width=\\textwidth}")
    latex_table = latex_table.replace("label{tab:test_agg_dataset}",
                                      "label{tab:test_agg_dataset}")

    latex_table = latex_table.replace("{c", "{c|").replace("c}", "|c}")
    latex_table = latex_table.replace("\\end{tabular}",
                                      "\\end{tabular}\\end{adjustbox}")
    latex_table = latex_table.replace("\\n        Average",
                        "\\n\\midrule        Average")
    with open(str(buf_name.absolute()), 'w') as f:
        f.write(latex_table)

parent_path = Path(__file__).resolve().parent.parent
base_path = parent_path / Path('./TABLE')


deep_methods = ['EEGNet-8,2', 'EEGTCNet', 'EEGITNet', 'EEGNeX', 'ShallowConvNet', 'DeepConvNet']
riemannian_methods = ['TS + SVM', 'TS + EL', 'TS + LR', 'FgMDM',  'MDM', 'ACM + TS + SVM']
raw_methods = ['CSP + SVM','CSP + LDA',  'DLCSPauto + shLDA','FBCSP + SVM', 'TRCSP + LDA', 'LogVar + SVM', 'LogVar + LDA']



df = pd.read_csv(parent_path / f'DATA/results_ITR_SSVEP_calculated.csv')

create_latex_table(df, "ITR_SSVEP", base_path)