import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import seaborn as sns

root = Path(__file__).parent.parent
data_path = root / 'moabb_results'
plots_path = root / 'pictures' / 'moabb_results' / 'rainclouds'
# csvs = data_path.glob('*.csv')
palettes = json.load(open(root / 'Scripts' / 'colours.json', 'r'))
best_pipelines = {"MotorImagery_all":"TS + EL", "MotorImagery_rf":"TS + EL",
                  "LeftRightImagery":"TS + EL", "P300":"TS + SVM", "SSVEP":"TS + LR"}

def take_data_from_file(file_path):
    df = pd.read_csv(file_path)
    return df["score","time","subject","session","dataset","pipeline","paradigm"]

def make_variability_file(df):
    final_df = pd.DataFrame(columns=["std_zscore","time","session","dataset","pipeline","paradigm"])
    # Getting all the existing dataset in the results file
    all_datasets = df["dataset"].unique()
    for dataset in all_datasets:
        df_dataset = df[df["dataset"]==dataset]
        # Getting all the existing session in the corresponding dataset
        all_sessions = df_dataset["session"].unique()

        for session in all_sessions:
            df_session = df_dataset[df_dataset["session"]==session]
            # Getting all the existing pipelines for the corresponding session
            all_pipelines = df_session["pipeline"].unique()

            for pipeline in all_pipelines:
                df_pipeline = df_session[df_session["pipeline"]==pipeline]
                z_score = (df_pipeline["score"] - np.mean(df_pipeline["score"])/np.std(df_pipeline["score"]))
                
                final_df = pd.concat([final_df,pd.DataFrame([{"std_zscore":np.std(z_score),
                                                            "time":np.mean(df_pipeline["time"]),
                                                            "session":session,"dataset":dataset,"pipeline":pipeline,
                                                            "paradigm":df_pipeline["paradigm"].iloc[0]}])],
                                    ignore_index=True)
    
    return final_df

def save_variability_file():
    df_P300 = pd.read_csv(data_path / 'results_P300.csv')
    df_SSVEP = pd.read_csv(data_path / 'results_SSVEP.csv')
    df_rf = pd.read_csv(data_path / 'results_rf.csv')
    df_lhrh = pd.read_csv(data_path / 'results_lhrh.csv')
    df_all = pd.read_csv(data_path/'results_all.csv')    

    # df_MI = pd.concat([df_all,df_rf],ignore_index=True)
    # df_MI = df_MI.drop_duplicates()

    final_df_all = make_variability_file(df_all)
    final_df_rf = make_variability_file(df_rf)
    final_df_lhrh = make_variability_file(df_lhrh)
    final_df_P300 = make_variability_file(df_P300)
    final_df_SSVEP = make_variability_file(df_SSVEP)
    final_df_all = final_df_all.replace("FilterBankMotorImagery","MotorImagery")
    final_df_all = final_df_all.replace("MotorImagery","MotorImagery_all")
    final_df_rf = final_df_rf.replace("FilterBankMotorImagery","MotorImagery")
    final_df_rf = final_df_rf.replace("MotorImagery","MotorImagery_rf")
    final_df_lhrh = final_df_lhrh.replace("FilterBankMotorImagery","LeftRightImagery")
    final_df_P300 = final_df_P300.replace("XDAWNCov + TS + SVM","TS + SVM")
    final_df_P300 = final_df_P300.replace("XDAWNCov + MDM","MDM")
    final_df_SSVEP = final_df_SSVEP.replace("SSVEP_TS + SVM","TS + SVM")
    final_df_SSVEP = final_df_SSVEP.replace("SSVEP_TS + LR","TS + LR")
    final_df_SSVEP = final_df_SSVEP.replace("SSVEP_MDM","MDM")


    final_df_all.to_csv(data_path/"variability_all.csv")
    final_df_rf.to_csv(data_path/"variability_rf.csv")
    final_df_lhrh.to_csv(data_path/"variability_lhrh.csv")
    final_df_P300.to_csv(data_path/"variability_P300.csv")
    final_df_SSVEP.to_csv(data_path/"variability_SSVEP.csv")

def plot_by_classif(pipeline=None):
    df_all = pd.read_csv(data_path/"variability_all.csv")
    df_rf = pd.read_csv(data_path/"variability_rf.csv")
    df_lhrh = pd.read_csv(data_path/"variability_lhrh.csv")
    df_P300 = pd.read_csv(data_path/"variability_P300.csv")
    print(df_P300)
    df_SSVEP = pd.read_csv(data_path/"variability_SSVEP.csv")
    df = pd.concat([df_SSVEP,df_P300,df_lhrh,df_all,df_rf],ignore_index=True)

    if pipeline is not None:
        df = df[df["pipeline"]==pipeline]
        plt.figure(1)
        sns.lineplot(data=df,x="paradigm",y="std_zscore",linestyle='dotted')
        plt.title("Variability for the pipeline {0}".format(pipeline))
        plt.xticks(rotation=45,fontsize=5)
    else:
        plt.figure(1)
        sns.lineplot(data=df,x="pipeline",y="std_zscore",hue="paradigm",legend="full",linestyle='dotted')
        plt.title("Variability for the all the pipelines")
        plt.xticks(rotation=75,fontsize=5)
    
    plt.show()

def plot_raincloud(pipelines,best_pipeline=False):
    df_all = pd.read_csv(data_path/"variability_all.csv")
    df_rf = pd.read_csv(data_path/"variability_rf.csv")
    df_lhrh = pd.read_csv(data_path/"variability_lhrh.csv")
    df_P300 = pd.read_csv(data_path/"variability_P300.csv")
    df_SSVEP = pd.read_csv(data_path/"variability_SSVEP.csv")
    if best_pipeline:
        df_all = df_all[df_all["pipeline"]==best_pipelines["MotorImagery_all"]]
        df_rf = df_rf[df_rf["pipeline"]==best_pipelines["MotorImagery_rf"]]
        df_lhrh = df_lhrh[df_lhrh["pipeline"]==best_pipelines["LeftRightImagery"]]
        df_P300 = df_P300[df_P300["pipeline"]==best_pipelines["P300"]]
        df_SSVEP = df_SSVEP[df_SSVEP["pipeline"]==best_pipelines["SSVEP"]]

        df = pd.concat([df_SSVEP,df_lhrh,df_all,df_rf,df_P300],ignore_index=True)
        prefix = "best"
    else:
        df = pd.concat([df_SSVEP,df_lhrh,df_all,df_rf,df_P300],ignore_index=True)

        df = df[df['pipeline'].isin(pipelines)]
        prefix=""

    f, ax = plt.subplots(figsize=(8, 3))
    # Pipelines (agg across datasets)
    pt.RainCloud(
        data=df, y='std_zscore', x='pipeline',
        hue='paradigm',
        bw='scott',
        width_viol=.7, width_box=0.5,
        dodge=True, point_dodge=False, orient='h',
        linewidth=0, box_linewidth=0,
        alpha=0.7, palette="Set2",
        box_showfliers=False,
        ax=ax, pointplot=False,
        point_linestyles="none", linecolor='k',
        point_markers='D',
    )
    # ax.set_xlim(0, 1)
    plt.xlabel("Z-Score standard deviation")
    plt.title("Variability accross paradigm and pipelines")
    plt.savefig(plots_path / 'all_variability_pipelines_{0}.pdf'.format(prefix), bbox_inches='tight')
    plt.show()

save_variability_file()
plot_by_classif()
# plot_raincloud(["TS + SVM","MDM"])
# plot_raincloud(["TS + SVM","MDM"],True)
