from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

best_pipelines = {"MotorImagery":["ShallowConvNet","CSP + LDA","TS + EL"],
                  "P300":[]}

def take_data_from_file(file_path):
    df = pd.read_csv(file_path)
    return df["score","time","subject","session","dataset","pipeline","paradigm"]

def make_variability_file(df):
    final_df = pd.DataFrame(columns=["score","time","session","dataset","pipeline","paradigm"])
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
                
                final_df = pd.concat([final_df,pd.DataFrame([{"score":np.std(df_pipeline["score"]),
                                                            "time":np.mean(df_pipeline["time"]),
                                                            "session":session,"dataset":dataset,"pipeline":pipeline,
                                                            "paradigm":df_pipeline["paradigm"].iloc[0]}])],
                                    ignore_index=True)
    
    return final_df

def save_variability_file():
    file_path = "C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\moabb_results\\"
    df_all = pd.read_csv(file_path+"results_All.csv")
    df_lhrh = pd.read_csv(file_path+"results_lhrh.csv")
    df_rf = pd.read_csv(file_path+"results_rf.csv")
    df_P300 = pd.read_csv(file_path+"results_P300.csv")
    df_SSVEP = pd.read_csv(file_path+"results_SSVEP.csv")

    df_MI = pd.concat([df_all,df_rf],ignore_index=True)
    df_MI = df_MI.drop_duplicates()

    final_df_MI = make_variability_file(df_MI)
    final_df_lhrh = make_variability_file(df_lhrh)
    final_df_P300 = make_variability_file(df_P300)
    final_df_SSVEP = make_variability_file(df_SSVEP)

    final_df_MI.to_csv(file_path+"variability_MI.csv")
    final_df_lhrh.to_csv(file_path+"variability_lhrh.csv")
    final_df_P300.to_csv(file_path+"variability_P300.csv")
    final_df_SSVEP.to_csv(file_path+"variability_SSVEP.csv")

def plot_by_classif(pipeline=None):
    file_path = "C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\moabb_results\\"
    df_MI = pd.read_csv(file_path+"variability_MI.csv")
    df_lhrh = pd.read_csv(file_path+"variability_lhrh.csv")
    df_P300 = pd.read_csv(file_path+"variability_P300.csv")
    df_SSVEP = pd.read_csv(file_path+"variability_SSVEP.csv")
    df = pd.concat([df_SSVEP,df_P300,df_lhrh,df_MI],ignore_index=True)

    if pipeline is not None:
        df = df[df["pipeline"]==pipeline]
        plt.figure(1)
        sns.lineplot(data=df,x="paradigm",y="score",linestyle='dotted')
        plt.title("Variability for the pipeline {0}".format(pipeline))
        plt.xticks(rotation=45,fontsize=5)
    else:
        plt.figure(1)
        sns.lineplot(data=df,x="pipeline",y="score",hue="paradigm",legend="full",linestyle='dotted')
        plt.title("Variability for the all the pipelines")
        plt.xticks(rotation=75,fontsize=5)
    
    plt.show()
    

# save_variability_file()
plot_by_classif("MDM")

