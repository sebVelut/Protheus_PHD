from os import listdir
from os.path import isfile,join
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt
import pandas as pd


HOME_PATH = "C:/Users/s.velut/Documents/These/Protheus_PHD/results/results/"


def plot_from_numpy(numpy,fig=None,axes=None):
    
    return fig, axes

def numpy2df(data,columns):
    if len(data.shape)!=2:
        ValueError("The data should be a 2dimension matrix")
    if data.shape[1]!=len(columns):
        ValueError("The number of columns of data should be the same as the number of values in columns")

    df = pd.DataFrame(data=data,columns=columns)


def oneDataframe(list_path,list_clf,save_path=None):
    list_numpy = []
    df = pd.DataFrame(columns=["index_participant","score","classifier","TF_method"])
    ind=0

    for p,clf in zip(list_path,list_clf):
        onlyfiles = [f for f in listdir(p) if isfile(join(p, f))]
        for f in onlyfiles:
            temp_np = np.load(p+f)
            temp_tf = str.split(f,"_")[0]

            for i,score in enumerate(temp_np):
                temp_row = [i+1,score,clf,temp_tf]
                df.loc[ind] = temp_row
                ind+=1
    
    if save_path is not None:
        df.to_csv(save_path,sep=";")
    
    return df

def median_comparaison(df,plots_path,prefix,spec=""):
    clfs = np.unique(df["classifier"])
    method = np.unique(df["TF_method"])
    plt.figure()
    
    ax = sns.barplot(data=df, x="classifier", y="score", hue="TF_method", estimator=np.median,saturation=0.6)
    

    ax.set_title('Accuracy score')
    ax.grid(color='#cccccc')
    ax.set_ylabel('score')
    ax.set_xlabel(None)
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.7)
    ax.minorticks_on()
    ax.set_xticklabels(df["classifier"].unique(), rotation='vertical')

    # plt.savefig(plots_path + 'bar_plot_score_{0}{1}.png'.format(prefix,spec), bbox_inches='tight')
    plt.show()

    return 0


def plot_score_rain(df,plots_path,prefix,spec=""):

    f, ax = plt.subplots(1,1)
    # Pipelines (agg across datasets)
    pt.RainCloud(
        data=df, y='score', x='classifier',
        hue='TF_method',
        bw='scott',
        width_viol=.7, width_box=0.5,
        dodge=True, point_dodge=False, orient='h',
        linewidth=0, box_linewidth=0,
        alpha=0.7, palette="Set2",
        box_showfliers=False,
        ax=ax, pointplot=False,
        point_linestyles="none", linecolor='k',
        point_markers='D',
        box_medianprops={"zorder": 11}
    )
    # ax.set_xlim(0, 1)
    plt.xlabel(prefix)
    plt.title("Accuracy of transfer learning method according to different pipelines")
    # plt.savefig(plots_path + 'score_{0}{1}.png'.format(prefix,spec), bbox_inches='tight')
    plt.show()

path_tf = HOME_PATH+"Score_TF/"
# df_score_code = oneDataframe([path_tf+"Xdawn_LDA/score_code/",path_tf+"CNN/score_code/",path_tf+"SPDNet/score_code/",
#                               path_tf+"TSMNet/score_code/",path_tf+"SPDbnNet/score_code/",path_tf+"BNSPDNet/score_code/"],
#                               ["Xdawn_LDA","CNN","SPDNet","TSMNet","SPDbnNet","BNSPDNet"],path_tf+"score_code.csv")
# plot_score_rain(df_score_code,path_tf,"score_code",spec="_all")
# df_score = oneDataframe([path_tf+"Xdawn_LDA/score/",path_tf+"CNN/score/",path_tf+"SPDNet/score/",
#                               path_tf+"TSMNet/score/",path_tf+"SPDbnNet/score/",path_tf+"BNSPDNet/score/"],
#                               ["Xdawn_LDA","CNN","SPDNet","TSMNet","SPDbnNet","BNSPDNet"],path_tf+"score.csv")
# plot_score_rain(df_score,path_tf,"score",spec="_all")
# df_tps_train_code = oneDataframe([path_tf+"Xdawn_LDA/temps_train_code/",path_tf+"CNN/temps_train_code/",
#                                   path_tf+"SPDNet/temps_train_code/",path_tf+"SPDbnNet/temps_train_code/",path_tf+"BNSPDNet/temps_train_code/"],
#                            ["Xdawn_LDA","CNN","SPDNet","SPDbnNet","BNSPDNet"],path_tf+"tps_train_code.csv")
# plot_score_rain(df_tps_train_code,path_tf,"tps_train_code",spec="_all")
# df_tps_test_code = oneDataframe([path_tf+"Xdawn_LDA/temps_test_code/",path_tf+"CNN/temps_test_code/",
#                                  path_tf+"SPDNet/temps_test_code/",path_tf+"SPDbnNet/temps_test_code/",path_tf+"BNSPDNet/temps_test_code/"],
#                            ["Xdawn_LDA","CNN","SPDNet","SPDbnNet","BNSPDNet"],path_tf+"tps_test_code.csv")
# plot_score_rain(df_tps_test_code,path_tf,"tps_test_code",spec="_all")



df_score_code = oneDataframe([path_tf+"CNN/score_code/",path_tf+"SPDNet/score_code/",path_tf+"SPDbnNet/score_code/"],
                              ["CNN","SPDNet","SPDbnNet"],path_tf+"score_code.csv")
plot_score_rain(df_score_code,path_tf,"score_code",spec="_reduce")
median_comparaison(df_score_code,path_tf,"score_code",spec="_reduce")
df_score = oneDataframe([path_tf+"CNN/score/",path_tf+"SPDNet/score/",path_tf+"SPDbnNet/score/"],
                              ["CNN","SPDNet","SPDbnNet"],path_tf+"score.csv")
plot_score_rain(df_score,path_tf,"score",spec="_reduce")
median_comparaison(df_score,path_tf,"score",spec="_reduce")
df_temps_test_code = oneDataframe([path_tf+"CNN/temps_test_code/",path_tf+"SPDNet/temps_test_code/",path_tf+"SPDbnNet/temps_test_code/"],
                              ["CNN","SPDNet","SPDbnNet"],path_tf+"temps_test_code.csv")
plot_score_rain(df_temps_test_code,path_tf,"temps_test_code",spec="_reduce")
median_comparaison(df_temps_test_code,path_tf,"temps_test_code",spec="_reduce")
df_temps_train_code = oneDataframe([path_tf+"CNN/temps_train_code/",path_tf+"SPDNet/temps_train_code/",path_tf+"SPDbnNet/temps_train_code/"],
                              ["CNN","SPDNet","SPDbnNet"],path_tf+"temps_train_code.csv")
plot_score_rain(df_temps_train_code,path_tf,"temps_train_code",spec="_reduce")
median_comparaison(df_temps_train_code,path_tf,"temps_train_code",spec="_reduce")
