######################################################################################################################
# Code to generate the figure for the Paper
######################################################################################################################
from pathlib import Path
from math import ceil
import json
import matplotlib.pyplot as plt
import PtitPrince as pt
import seaborn as sns
import numpy as np
import pandas as pd
from os.path import dirname, join


def get_pipelines_order(df):
    # Rank the pipelines within each session:
    df['rank_in_session'] = df.groupby(['dataset', 'subject', 'session',]).score.rank()
    # Aggregate the ranks:
    return df.groupby(['dataset','pipeline','pipeline_category']
            ).rank_in_session.mean().reset_index( # Aggregate ranks by average within each dataset
            ).groupby(['pipeline','pipeline_category']
            ).rank_in_session.mean().sort_values(ascending=False).reset_index() # Aggregate ranks by average across dataset


root = Path(__file__).parent.parent
# root = Path(".").cwd().parent
data_path = root / 'article ITR'
plots_path = root / 'article ITR' 

# Load Data and Palette
p300_df = pd.read_csv(data_path / 'results_P300_all.csv',sep=';')
ssvep_df = pd.read_csv(data_path / 'results_ITR_SSVEP_calculated.csv',sep=',')
mi_rf_df = pd.read_csv(data_path / 'results_rf_Optuna.csv',sep=',')
mi_lhrh_df = pd.read_csv(data_path / 'results_lhrh_Optuna.csv',sep=',')
mi_all_df = pd.read_csv(data_path / 'results_All_Optuna.csv',sep=',')
palettes = json.load(open(root / 'article ITR' / 'colours.json', 'r'))
sns.set(font="serif", style="whitegrid", palette='Set2', color_codes=False)
# Example usage
#sns.boxplot(..., hue='pipeline_category', palette=palettes['pipeline_category'],...)

# Set the plotting options
boxplot_lw = 1.0
boxplot_props = {'linewidth': boxplot_lw}

# Order Pipeline by score
dfs = dict(
    mi_all=mi_all_df,
    mi_rf=mi_rf_df,
    mi_lhrh=mi_lhrh_df,
    p300=p300_df,
    ssvep=ssvep_df,
)

# Count the number of sessions in each dataset:
dataset_counts = {}
for name, df in dfs.items():
    print(name)
    dataset_counts[name] = df.groupby((['dataset', 'pipeline'])).score.count().reset_index().groupby('dataset').score.max()
    print(name,'\n', dataset_counts[name])
mi_pipelines_order = get_pipelines_order(pd.concat([df for n, df in dfs.items() if 'mi' in n]))

# Code Order
for name, df in dfs.items():
    if 'mi' in name:
        df.pipeline = df.pipeline.astype(pd.CategoricalDtype(categories=mi_pipelines_order.pipeline.to_list()))
    else:
        pipelines_order = get_pipelines_order(df)
        df.pipeline = df.pipeline.astype(pd.CategoricalDtype(categories=pipelines_order.pipeline.to_list()))
    # df['pipeline_category'] = df['pipeline_category'].astype('category')
    dataset_order = df.groupby('dataset').score.mean().sort_values(ascending=False)
    df.dataset = df.dataset.astype(pd.CategoricalDtype(categories=dataset_order.index.to_list()))

#######################################################################################################################
db_labels_map = {'BNCI2014-001': 'BNCI\n2014-001', 'BNCI2014-002': 'BNCI\n2014-002', 'BNCI2014-004': 'BNCI\n2014-004',
                 'Cho2017': 'Cho\n2017', 'GrosseWentrup2009': 'GrosseWentrup\n2009', 'Lee2019-MI': 'Lee\n2019-MI',
                 'PhysionetMotorImagery': 'Physionet\nMotorImagery', 'Schirrmeister2017': 'Schirrmeister\n2017',
                 'Shin2017A': 'Shin\n2017A', 'Weibo2014': 'Weibo\n2014', 'Zhou2016': 'Zhou\n2016', 'Average': 'Average\n',
                 'AlexandreMotorImagery': 'Alexandre\nMotorImagery', 'BNCI2015-001': 'BNCI\n2015-001',
                 'BNCI2015-004': 'BNCI\n2015-004',
                 'BNCI2014-009': 'BNCI\n2014-009', 'Huebner2017': 'Huebner\n2017', 'Huebner2018': 'Huebner\n2018',
                 'BNCI2015-003': 'BNCI\n2015-003', 'EPFLP300': 'EPFL\nP300', 'BrainInvaders2014a': 'BI\n2014a',
                 'BNCI2014-008': 'BNCI\n2014-008', 'Cattan2019-VR': 'Cattan\n2019-VR',
                 'BrainInvaders2014b': 'BI\n2014b', 'Lee2019-ERP': 'Lee\n2019-ERP',
                 'Sosulski2019': 'Sosulski\n2019', 'BrainInvaders2013a': 'BI\n2013a',
                 'BrainInvaders2015a': 'BI\n2015a', 'BrainInvaders2015b': 'BI\n2015b',
                 'BrainInvaders2012': 'BI\n2012'}

pip_labels_map = {'ShallowConvNet': 'Shallow\nConvNet', 'EEGNet-8.2': 'EEG\nNet-8,2', 'DeepConvNet': 'Deep\nConvNet',
                  'DLCSPauto + shLDA': 'DLCSPauto\nshLDA',
                  'EEGTCNet': 'EEG\nTCNet', 'EEGITNet': 'EEG\nITNet', 'EEGNeX': 'EEG\nNeX',
                  'SSVEP_TS + LR': 'SSVEP\nTS+LR', 'SSVEP_TS + SVM': 'SSVEP\nTS+SVM', 'SSVEP_MDM': 'SSVEP\nMDM',
                  'TRCA': 'TRCA\n', 'TRCSP + LDA': 'TRCSP\nLDA', 'MsetCCA': 'Mset\nCCA', 'CCA': 'CCA\n',
                  'TS + EL': 'TS\nEL', 'TS + SVM': 'TS+SVM\n', 'TS + LR': 'TS\nLR',
                  'ACM + TS + SVM': 'ACM\nTS\nSVM', 'FgMDM': 'FgMDM\n', 'MDM': 'MDM\n', 'CSP + SVM': '\nCSP\nSVM',
                  'CSP + LDA': '\nCSP\nLDA', 'FBCSP + SVM': '\nFBCSP\nSVM'}
paradigms_map = {'p300': 'P300', 'ssvep': 'SSVEP', 'mi_all': 'MI All', 'mi_rf': 'MI RH-F', 'mi_lhrh': 'MI RH-LH'}
# #######################################################################################################################
# # MI : best DL is shallowConvNet (without augmentation) (Figure5a)
# df2 = mi_lhrh_df[mi_lhrh_df['pipeline_category'] == "Deep Learning"].copy()
# df2.pipeline = df2.pipeline.cat.remove_unused_categories()

# n_pi = df2.pipeline.unique().shape[0]

# palette_5a = dict(zip(['ShallowConvNet', 'EEGNet-8.2', 'DeepConvNet', 'EEGTCNet', 'EEGITNet', 'EEGNeX'], sns.color_palette('Set2', 6)))

# f, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, n_pi+2))
# pt.RainCloud(
#     data=df2, y='score', x='pipeline',
#     hue=None,
#     bw='scott',
#     width_viol=.7, width_box=0.3,
#     dodge=False, point_dodge=False, orient='h',
#     linewidth=0, box_linewidth=boxplot_lw,
#     box_whiskerprops=boxplot_props,
#     box_medianprops=boxplot_props,
#     alpha=0.7, palette=palette_5a,
#     box_showfliers=False,
#     ax=ax[0][0], pointplot=True,
#     point_linestyles="none", linecolor='k',
#     point_markers='D',)

# ax[0][0].set_yticks([])

# for i in range(n_pi):
#     ax[0][0].axhline(y=i + 0.3, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax[0][0].set_xlim(right=1.02)

# ax[0][0].set(ylabel="Deep Learning pipelines", xlabel="ROC-AUC score\nMI RH-LH datasets\n(a)")
# ax[0][0].legend(['ShallowConvNet', 'EEGNet-8,2', 'DeepConvNet', 'EEGTCNet', 'EEGITNet', 'EEGNeX'],
#              title='Deep Learning pipeline',
#              loc=(0.1, 1.01), ncols=n_pi)
# ########################################################################################################################
# # MI : #trial influence → deep need many trials, few trial = best acc w Riem/Raw ONLY DL (Figure5b)
# samples_map = {"RHF": {"(0, 50]": [0, 50], "(50, 150]": [50, 150], ">150": [150, 10000]}}
# samples_cat = np.asarray(mi_rf_df["samples"].copy())
# for idx, d in enumerate(samples_map["RHF"]):
#     samples_cat[np.logical_and(samples_map["RHF"][d][0] < samples_cat,
#                                samples_cat <= samples_map["RHF"][d][1])] = idx
# mi_lhrh_df_2 = pd.read_csv(data_path / 'results_rf_Optuna.csv')
# mi_lhrh_df_2["samples_cat"] = samples_cat

# p_ord = list(np.arange(len(samples_map["RHF"].keys())))

# sns.boxplot(data=mi_lhrh_df_2[mi_lhrh_df_2["pipeline_category"] == 'Deep Learning'],
#             y="samples_cat",
#             x="score",
#             hue="pipeline",
#             linewidth=boxplot_lw,
#             whiskerprops=boxplot_props,
#             medianprops=boxplot_props,
#             hue_order=['ShallowConvNet', 'EEGNet-8.2', 'DeepConvNet', 'EEGTCNet', 'EEGITNet', 'EEGNeX'],
#             palette=palette_5a,
#             ax=ax[0][1],
#             boxprops=dict(alpha=0.7),
#             orient='h')

# sns.stripplot(data=mi_lhrh_df_2[mi_lhrh_df_2["pipeline_category"] == "Deep Learning"],
#               y="samples_cat",
#               x="score",
#               hue="pipeline",
#               hue_order=['ShallowConvNet', 'EEGNet-8.2', 'DeepConvNet', 'EEGTCNet', 'EEGITNet', 'EEGNeX'],
#               ax=ax[0][1],
#               dodge=True,
#               linewidth=0,
#               palette=palette_5a,
#               edgecolor='k',
#               size=3,
#               zorder=0,
#               orient='h')

# ax[0][1].yaxis.set_label_position("right")
# ax[0][1].yaxis.tick_right()
# ax[0][1].set_yticklabels([l for l in samples_map["RHF"].keys()], rotation=90, va='center')

# for i in range(len(p_ord)):
#     ax[0][1].axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax[0][1].set_xlim([0.15, 1.02])

# ax[0][1].set(ylabel="Number of samples", xlabel="ROC-AUC score\nMI RH-F datasets\n(b)")
# ax[0][1].legend().set_visible(False)

# plt.subplots_adjust(wspace=0.05)
# # plt.savefig(plots_path / 'Figure5a_mi_rhlh_b_mi_rhf_deep_learning.pdf', bbox_inches='tight')
# ######################################################################################################################
# # SSVEP a : (Figure5a)
# df2 = ssvep_df[ssvep_df['pipeline_category'] == "Deep Learning"].copy()
# df2.pipeline = df2.pipeline.cat.remove_unused_categories()
# n_pi = df2.pipeline.unique().shape[0]

# # f, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3 * 1.5))

# pt.RainCloud(
#     data=df2, y='ITR', x='pipeline',
#     hue=None,
#     bw='scott',
#     width_viol=.7, width_box=0.3,
#     dodge=False, point_dodge=False, orient='h',
#     linewidth=0, box_linewidth=boxplot_lw,
#     box_whiskerprops=boxplot_props,
#     box_medianprops=boxplot_props,
#     alpha=0.7, palette=palette_5a,
#     box_showfliers=False,
#     ax=ax[1][0], pointplot=True,
#     point_linestyles="none", linecolor='k',
#     point_markers='D',)

# ax[1][0].set_yticks([])

# for i in range(n_pi):
#     ax[1][0].axhline(y=i + 0.3, xmin=0, xmax=1.5, color='black', linestyle=':', alpha=0.3)
# ax[1][0].set_xlim(left=0)

# ax[1][0].set(ylabel="Deep Learning pipelines", xlabel="ITR score\nSSVEP datasets\n(c)")
# ax[1][0].legend().set_visible(False)
# # ax[1][0].legend(['ShallowConvNet', 'EEGNet-8,2', 'EEGITNet', 'EEGNeX'],
# #              title='Deep Learning pipeline',
# #              loc=(-0.1, 1.01), ncols=n_pi).set_visible(False)
# plt.subplots_adjust(hspace=0.4)
# ########################################################################################################################
# # SSVEP b : (Figure5b)
# # samples_map = {"SSVEP": {"(0, 75]": [0, 75], "(75, 125]": [75, 125], ">125": [125, 10000]}}
# # samples_cat = np.asarray(ssvep_df["samples"].copy())
# # for idx, d in enumerate(samples_map["SSVEP"]):
# #     samples_cat[np.logical_and(samples_map["SSVEP"][d][0] < samples_cat,
# #                                samples_cat <= samples_map["SSVEP"][d][1])] = idx
# # ssvep_df_2 = pd.read_csv(data_path / 'results_SSVEP_all.csv',sep=";")
# # ssvep_df_2["samples_cat"] = samples_cat
# # p_ord = list(np.arange(len(samples_map["SSVEP"].keys())))

# # sns.boxplot(data=ssvep_df_2[ssvep_df_2["pipeline_category"] == 'Deep Learning'],
# #             y="samples_cat",
# #             x="score",
# #             hue="pipeline",
# #             linewidth=boxplot_lw,
# #             whiskerprops=boxplot_props,
# #             medianprops=boxplot_props,
# #             hue_order=['ShallowConvNet', 'EEGNet-8.2', 'EEGITNet', 'EEGNeX'],
# #             palette="Set2",
# #             ax=ax[1],
# #             boxprops=dict(alpha=0.7),
# #             orient='h')

# # sns.stripplot(data=ssvep_df_2[ssvep_df_2["pipeline_category"] == "Deep Learning"],
# #               y="samples_cat",
# #               x="score",
# #               hue="pipeline",
# #               hue_order=['ShallowConvNet', 'EEGNet-8.2', 'EEGITNet', 'EEGNeX'],
# #               ax=ax[1],
# #               dodge=True,
# #               linewidth=0,
# #               palette="Set2",
# #               edgecolor='k',
# #               size=3,
# #               zorder=0,
# #               orient='h')

# # ax[1].yaxis.set_label_position("right")
# # ax[1].yaxis.tick_right()
# # ax[1].set_yticklabels([l for l in samples_map["SSVEP"].keys()], rotation=90, va='center')

# # for i in range(len(p_ord)):
# #     ax[1].axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# # ax[1].set_xlim([0.15, 1.02])

# # ax[1].set(ylabel="Number of samples", xlabel="ITR score\nSSVEP datasets\n(b)")
# # ax[1].legend().set_visible(False)

# # plt.subplots_adjust(wspace=0.05)
# # plt.savefig(plots_path / 'Figure5a_ssvep_deep_learning.pdf', bbox_inches='tight')
# ######################################################################################################################
# # ERP a : (Figure5a)
# df2 = p300_df[p300_df['pipeline_category'] == "Deep Learning"].copy()
# df2.pipeline = df2.pipeline.cat.remove_unused_categories()
# n_pi = df2.pipeline.unique().shape[0]

# # f, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, n_pi))

# # pt.RainCloud(
# #     data=df2, y='score', x='pipeline',
# #     hue=None,
# #     bw='scott',
# #     width_viol=.7, width_box=0.3,
# #     dodge=False, point_dodge=False, orient='h',
# #     linewidth=0, box_linewidth=boxplot_lw,
# #     box_whiskerprops=boxplot_props,
# #     box_medianprops=boxplot_props,
# #     hue_order=['ShallowConvNet', 'EEGNet-8.2', 'EEGITNet', 'EEGNeX'],
# #     alpha=0.7, palette="Set2",
# #     box_showfliers=False,
# #     ax=ax[0], pointplot=True,
# #     point_linestyles="none", linecolor='k',
# #     point_markers='D',)

# # ax[0].set_yticks([])

# # for i in range(n_pi):
# #     ax[0].axhline(y=i + 0.3, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# # ax[0].set_xlim(right=1.02)

# # ax[0].set(ylabel="Deep Learning pipelines", xlabel="ROC-AUC score\nP300 datasets\n(d)")
# # ax[0].legend(['ShallowConvNet', 'EEGNet-8,2', 'EEGITNet', 'EEGNeX'],
# #              title='Deep Learning pipeline',
# #              loc=(0.1, 1.01), ncols=n_pi)
# ########################################################################################################################
# # ERP b : (Figure5b)
# samples_map = {"RHF": {"(0, 500]": [0, 500], "(500, 1500]": [500, 1500], ">1500": [1500, 100000]}}
# samples_cat = np.asarray(p300_df["samples"].copy())
# for idx, d in enumerate(samples_map["RHF"]):
#     samples_cat[np.logical_and(samples_map["RHF"][d][0] < samples_cat,
#                                samples_cat <= samples_map["RHF"][d][1])] = idx
# p300_df_2 = pd.read_csv(data_path / 'results_P300_all.csv',sep=";")
# p300_df_2["samples_cat"] = samples_cat

# p_ord = list(np.arange(len(samples_map["RHF"].keys())))

# sns.boxplot(data=p300_df_2[p300_df_2["pipeline_category"] == 'Deep Learning'],
#             y="samples_cat",
#             x="score",
#             hue="pipeline",
#             linewidth=boxplot_lw,
#             whiskerprops=boxplot_props,
#             medianprops=boxplot_props,
#             hue_order=['ShallowConvNet', 'EEGNet-8.2', 'EEGITNet', 'EEGNeX'],
#             palette=palette_5a,
#             ax=ax[1][1],
#             boxprops=dict(alpha=0.7),
#             orient='h')

# sns.stripplot(data=p300_df_2[p300_df_2["pipeline_category"] == "Deep Learning"],
#               y="samples_cat",
#               x="score",
#               hue="pipeline",
#               hue_order=['ShallowConvNet', 'EEGNet-8.2', 'EEGITNet', 'EEGNeX'],
#               ax=ax[1][1],
#               dodge=True,
#               linewidth=0,
#               palette="Set2",
#               edgecolor='k',
#               size=3,
#               zorder=0,
#               orient='h')

# ax[1][1].yaxis.set_label_position("right")
# ax[1][1].yaxis.tick_right()
# ax[1][1].set_yticklabels([l for l in samples_map["RHF"].keys()], rotation=90, va='center')

# for i in range(len(p_ord)):
#     ax[1][1].axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax[1][1].set_xlim([0.15, 1.02])

# ax[1][1].set(ylabel="Number of samples", xlabel="ROC-AUC score\nP300 datasets\n(d)")
# ax[1][1].legend().set_visible(False)

# plt.subplots_adjust(wspace=0.05)
# # plt.savefig(plots_path / 'Figure5a_p300ab_deep_learning.pdf', bbox_inches='tight')
# plt.savefig(plots_path / 'Figure5a_all.pdf', bbox_inches='tight')
# ######################################################################################################################
# #######################################################################################################################
# # MI : #trial influence → deep need many trials, few trial = best acc w Riem/Raw (Figure3)
# samples_map = {"RHF": {"(0, 50]": [0, 50],
#                        "(50, 150]": [50, 150],
#                        ">150": [150, 10000]}}
# yrange = {"RHF": [0.15, 1.02]}

# samples_cat = np.asarray(mi_rf_df["samples"].copy())
# for idx, d in enumerate(samples_map["RHF"]):
#     samples_cat[np.logical_and(samples_map["RHF"][d][0] < samples_cat,
#                                samples_cat <= samples_map["RHF"][d][1])] = idx

# mi_rf_df["samples_cat"] = samples_cat
# p_ord = list(np.arange(len(samples_map["RHF"].keys())))

# fig, ax = plt.subplots(1, 1, figsize=(7, 3 * 1.5))

# sns.boxplot(data=mi_rf_df,
#             y="samples_cat",
#             x="score",
#             hue="pipeline_category",
#             order=p_ord,
#             palette=palettes['pipeline_category'],
#             ax=ax,
#             boxprops=dict(alpha=0.7),
#             orient='h')

# sns.stripplot(data=mi_rf_df,
#               y="samples_cat",
#               x="score",
#               hue="pipeline_category",
#               ax=ax,
#               dodge=True,
#               linewidth=0,
#               edgecolor='k',
#               palette=palettes['pipeline_category'],
#               size=3,
#               zorder=0,
#               orient='h')

# ytick_labels = [l + '\n' for l in samples_map["RHF"].keys()]
# ax.set_yticklabels(ytick_labels, rotation=90, va='center')
# handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles=handles[0:len(handles) // 2], title='Pipeline category',
#            loc=(0.0, 1.01), ncols=len(samples_map["RHF"]))
# ax.set(ylabel="Number of samples", xlabel="ROC-AUC score")
# for i in range(len(ytick_labels)):
#     plt.axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax.set_xlim(yrange["RHF"])
# plt.savefig(plots_path / 'Figure3_mi_rh_f_scores_vs_epochs.pdf', bbox_inches='tight')
# #######################################################################################################################
# # SSVEP (Figure3)
# samples_map = {"RHF": {"(0, 75]": [0, 75], "(75, 125]": [75, 125], ">125": [125, 10000]}}
# yrange = {"RHF": [0.15, 1.02]}

# samples_cat = np.asarray(ssvep_df["samples"].copy())
# for idx, d in enumerate(samples_map["RHF"]):
#     samples_cat[np.logical_and(samples_map["RHF"][d][0] < samples_cat,
#                                samples_cat <= samples_map["RHF"][d][1])] = idx

# ssvep_df["samples_cat"] = samples_cat
# p_ord = list(np.arange(len(samples_map["RHF"].keys())))

# fig, ax = plt.subplots(1, 1, figsize=(7, 3 * 1.5))

# sns.boxplot(data=ssvep_df,
#             y="samples_cat",
#             x="score",
#             hue="pipeline_category",
#             order=p_ord,
#             palette=palettes['pipeline_category'],
#             ax=ax,
#             boxprops=dict(alpha=0.7),
#             orient='h')

# sns.stripplot(data=ssvep_df,
#               y="samples_cat",
#               x="score",
#               hue="pipeline_category",
#               ax=ax,
#               dodge=True,
#               linewidth=0,
#               edgecolor='k',
#               palette=palettes['pipeline_category'],
#               size=3,
#               zorder=0,
#               orient='h')

# ytick_labels = [l + '\n' for l in samples_map["RHF"].keys()]
# ax.set_yticklabels(ytick_labels, rotation=90, va='center')
# handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles=handles[0:len(handles) // 2], title='Pipeline category',
#            loc=(0.0, 1.01), ncols=len(samples_map["RHF"]))
# ax.set(ylabel="Number of samples", xlabel="Accuracy score")
# for i in range(len(ytick_labels)):
#     plt.axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# # ax.set_xlim(yrange["RHF"])
# plt.savefig(plots_path / 'Figure3_ssvep_scores_vs_epochs.pdf', bbox_inches='tight')
# #######################################################################################################################
# # P300 (Figure3)
# samples_map = {"RHF": {"(0, 500]": [0, 500],
#                        "(500, 1500]": [500, 1500],
#                        ">1500": [1500, 10000]}}
# yrange = {"RHF": [0.15, 1.02]}

# samples_cat = np.asarray(p300_df["samples"].copy())
# for idx, d in enumerate(samples_map["RHF"]):
#     samples_cat[np.logical_and(samples_map["RHF"][d][0] < samples_cat,
#                                samples_cat <= samples_map["RHF"][d][1])] = idx

# p300_df["samples_cat"] = samples_cat
# p_ord = list(np.arange(len(samples_map["RHF"].keys())))

# fig, ax = plt.subplots(1, 1, figsize=(7, 3 * 1.5))

# sns.boxplot(data=p300_df,
#             y="samples_cat",
#             x="score",
#             hue="pipeline_category",
#             order=p_ord,
#             palette=palettes['pipeline_category'],
#             ax=ax,
#             boxprops=dict(alpha=0.7),
#             orient='h')

# sns.stripplot(data=p300_df,
#               y="samples_cat",
#               x="score",
#               hue="pipeline_category",
#               ax=ax,
#               dodge=True,
#               linewidth=0,
#               edgecolor='k',
#               palette=palettes['pipeline_category'],
#               size=3,
#               zorder=0,
#               orient='h')

# ytick_labels = [l + '\n' for l in samples_map["RHF"].keys()]
# ax.set_yticklabels(ytick_labels, rotation=90, va='center')
# handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles=handles[0:len(handles) // 2], title='Pipeline category',
#            loc=(0.0, 1.01), ncols=len(samples_map["RHF"]))
# ax.set(ylabel="Number of samples", xlabel="ROC-AUC score")
# for i in range(len(ytick_labels)):
#     plt.axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax.set_xlim(yrange["RHF"])
# plt.savefig(plots_path / 'Figure3_p300_scores_vs_epochs.pdf', bbox_inches='tight')
# #######################################################################################################################
# # MI : #chan : small regime : Riem++, large : DL + (Figure4a)
# samples_map = {"RHLF": {"(0, 25]": [0, 25], "(25, 80]": [25, 90], ">80": [80, 150]}}
# samples_cat = np.asarray(mi_lhrh_df["channels"])
# for idx, d in enumerate(samples_map["RHLF"]):
#     samples_cat[np.logical_and(samples_map["RHLF"][d][0] < samples_cat,
#                                samples_cat <= samples_map["RHLF"][d][1])] = idx
# mi_lhrh_df["samples_cat"] = samples_cat
# p_ord = list(np.arange(len(samples_map["RHLF"].keys())))

# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 3 * 2))

# sns.boxplot(data=mi_lhrh_df,
#             y="samples_cat",
#             x="score",
#             hue="pipeline_category",
#             palette=palettes['pipeline_category'],
#             order=p_ord,
#             ax=ax[0],
#             linewidth=boxplot_lw,
#             boxprops=dict(alpha=0.7),
#             orient='h')

# sns.stripplot(data=mi_lhrh_df,
#               y="samples_cat",
#               x="score",
#               hue="pipeline_category",
#               ax=ax[0],
#               dodge=True,
#               linewidth=0,
#               palette=palettes['pipeline_category'],
#               edgecolor='k',
#               size=3,
#               zorder=0,
#               orient='h')

# ax[0].set_yticklabels([l for l in samples_map["RHLF"].keys()], rotation=90, va='center')
# handles, labels = ax[0].get_legend_handles_labels()
# ax[0].legend(handles=handles[0:len(handles) // 2], title='Pipeline category', ncols=1)
# ax[0].set(ylabel="Number of channels", xlabel="ROC-AUC score\nMI RH-LH datasets\n(a)")
# for i in range(len(p_ord)):
#     ax[0].axhline(y=i + 0.5, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax[0].set_xlim([-0.05, 1.02])
# #######################################################################################################################
# # All : Riem TS = FgMDM > MDM (Figure4b)
# df2 = mi_lhrh_df[mi_lhrh_df['pipeline_category'] == "Riemannian"].copy()

# df2.pipeline = df2.pipeline.cat.remove_unused_categories()

# # Pipelines (agg across datasets)
# pt.RainCloud(
#     data=df2, y='score', x='pipeline',
#     bw='scott',
#     width_viol=.7, width_box=0.3,
#     dodge=False, point_dodge=False, orient='h',
#     linewidth=0, box_linewidth=boxplot_lw,
#     box_whiskerprops=boxplot_props,
#     box_medianprops=boxplot_props,
#     alpha=0.7, palette="Set2",
#     box_showfliers=False,
#     ax=ax[1], pointplot=True,
#     point_linestyles="none", linecolor='k',
#     point_markers='D',
# )

# ax[1].yaxis.set_label_position("right")
# ax[1].yaxis.tick_right()
# ytick_labels = [pip_labels_map[l] for l in [l.get_text() for l in ax[1].get_yticklabels()]]
# ytick_labels[0] = 'ACM+\nTS+SVM'
# ax[1].set_yticks([x - 0.25 for x in ax[1].get_yticks()])
# ax[1].set_yticklabels(ytick_labels, rotation=90, va='center', ma='center')

# for i in range(len(ytick_labels)):
#     ax[1].axhline(y=i + 0.3, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax[1].set_xlim(right=1.02)

# ax[1].set(ylabel="Riemannian pipelines", xlabel="ROC-AUC score\nMI RH-LH datasets\n(b)")
# plt.subplots_adjust(wspace=0.05)
# plt.savefig(plots_path / 'Figure4_mi_lhrh_a_no_channels_b_riemannian_agg.pdf', bbox_inches='tight')
# ######################################################################################################################
# # Rainplot only using the best 3 pipeline (Figure12)
# best_n = 3
# name = 'mi_rf'
# full_id = f'{name}_best{best_n}'

# mi_rhf_pipelines_order = get_pipelines_order(pd.concat([df for n, df in dfs.items() if 'mi_rf' in n]))

# best_pipes = mi_rhf_pipelines_order.groupby('pipeline_category').head(best_n)
# df2 = mi_rf_df.copy()[mi_rf_df.pipeline.isin(best_pipes.pipeline.to_list())]
# df2.pipeline = df2.pipeline.cat.remove_unused_categories()
# n_pi = df2.pipeline.unique().shape[0]
# print(n_pi)
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), width_ratios=[1, 1.15])

# ######################################################################################################################
# # Pipelines (agg across datasets)
# pt.RainCloud(
#     data=df2, y='score', x='pipeline',
#     hue='pipeline_category',
#     bw='scott',
#     width_viol=.7, width_box=0.5,
#     dodge=False, point_dodge=False, orient='h',
#     linewidth=0, box_linewidth=0,
#     alpha=0.7, palette=palettes['pipeline_category'],
#     box_showfliers=False,
#     ax=ax[0], pointplot=True,
#     point_linestyles="none", linecolor='k',
#     point_markers='D',
#     order=best_pipes.pipeline.to_list()
# )

# handles, labels = ax[0].get_legend_handles_labels()
# unique_labels = {}
# for handle, label in zip(handles, labels):
#     if label not in unique_labels:
#         unique_labels[label] = handle

# ytick_labels = [l.get_text() for l in ax[0].get_yticklabels()]
# ytick_labels = [pip_labels_map[l] for l in ytick_labels]
# ax[0].set_yticks([x - 0.25 for x in ax[0].get_yticks()])
# ax[0].set_yticklabels(ytick_labels, rotation=90, va='center', ma='center')
# for i in range(len(ytick_labels)):
#     ax[0].axhline(y=i + 0.3, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
# ax[0].set_xlim(right=1.02)
# ax[0].legend(unique_labels.values(), unique_labels.keys(), loc=(0.0, 1.01), ncols=3, title='Pipeline category')
# ax[0].set(ylabel=f"Best {best_n} MI RH-F pipelines in each category", xlabel="ROC-AUC score\nMI RH-F datasets")

# ######################################################################################################################
# # Pipeline rank figure (12b)
# df = dfs['mi_rf']
# gp = df.groupby(['dataset', 'subject', 'session'], dropna=True).score
# df['weight_session'] = 1
# df['rank_min'] = gp.rank(method='min', ascending=False)
# df['rank_max'] = gp.rank(method='max', ascending=False)
# print(f"{name} - count groups with a certain number of pipelines \n{gp.count().value_counts().sort_index()}")

# rows = []
# df['rank_in_session'] = 0
# for _, r in df.iterrows():
#     rmin = int(r['rank_min'])
#     rmax = int(r['rank_max'])
#     for j in range(rmin, rmax+1):
#         r1 = r.copy()
#         r1.weight_session = 1/(rmax-rmin+1)
#         r1.rank_in_session = int(j)
#         rows.append(r1)
# df = pd.DataFrame(rows)
# print(f"{name} - sum rank weights after \n{df.groupby('rank_in_session').weight_session.sum().sort_index()}")
# print(f"{name} - count ranks after \n{df.rank_in_session.value_counts().sort_index()}")
# df.pipeline = df.pipeline.astype(pd.CategoricalDtype(categories=mi_pipelines_order.pipeline.to_list()))
# df.pipeline = df.pipeline.cat.remove_unused_categories()

# n_pip = len(df.pipeline.unique())
# p1 = sns.color_palette('pastel', n_colors=ceil(n_pip/2))
# p2 = sns.color_palette('muted', n_colors=n_pip-len(p1))
# palette = [(p1 if i % 2 == 0 else p2)[i // 2] for i in range(n_pip)]
# n_ranks = int(df['rank_in_session'].max())
# print(len(palette))

# sns.histplot(data=df, x="rank_in_session", hue="pipeline", palette=palette,
#             ax=ax[1],
#             multiple="stack",weights='weight_session',#stat='probability',
#             element='step',
#             # discrete=True,
#             binwidth=1,
#             binrange=(0.5, n_ranks+0.5),
#             edgecolor='k')
# ax[1].set_xticks(range(1, n_ranks+1) if n_ranks < 10 else range(1, n_ranks+1, 2))
# legend_dict = {}
# for m_idx, m_name in enumerate(ax[1].get_legend().texts):
#     legend_dict[m_name.get_text()] = ax[1].get_legend().legend_handles[m_idx]
# ax[1].set_ylim([-0.05, 231.55])
# ax[1].set_xlim([0.45, 16.55])
# ax[1].grid(False)
# ax[1].set(xlabel="Pipeline rank", ylabel="Session count")
# ax[1].legend(legend_dict.values(), legend_dict.keys(), loc=(0.0, 1.01), ncols=4, title='Pipelines',
#              handletextpad=0.4, columnspacing=0.5, handlelength=1.25)
# plt.savefig(plots_path / f'Figure12_{full_id}_agg.pdf', bbox_inches='tight')
#####################################################################################################################
######################################################################################################################
# MI : execution time ? compare DL/Raw/Riemann (Figure6)
n_ds = mi_rf_df.dataset.unique().shape[0]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 3 * 1.5))

df_agg = mi_rf_df.groupby('dataset').agg({'time': 'mean'})
df_agg = df_agg.sort_values(by='time', ascending=True).reset_index()

# Categorize the results DataFrame based on the sorted df_agg
mi_rf_df_withaug = mi_rf_df.copy()
mi_rf_df_withaug['dataset'] = pd.Categorical(mi_rf_df_withaug['dataset'],
                                             categories=df_agg['dataset'],
                                             ordered=True)

mi_rf_df_withaug['pipeline_category'] = pd.Categorical(mi_rf_df_withaug['pipeline_category'],
                                                       categories=['Riemannian', 'Raw', 'Deep Learning'],
                                                       ordered=True, )

# Sort the results DataFrame by order
mi_rf_df_withaug.sort_values('dataset', inplace=True, ascending=True)

sns.barplot(data=mi_rf_df_withaug.reset_index(),
            y='pipeline_category',
            x='time',
            hue='pipeline_category',
            alpha=0.7,
            palette=palettes['pipeline_category'],
            log=True,
            ax=ax[0],
            orient='h')

ax[0].legend([], [], frameon=False)  # Remove legend of fig a)

ax[0].set_ylabel("MI RH-F datasets")
ax[0].set_xlabel("Execution time [s]")
handles, labels = ax[0].get_legend_handles_labels()
unique_labels = {}
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels[label] = handle

ytick_labels = [l.get_text() for l in ax[0].get_yticklabels()]
# ytick_labels = [db_labels_map[l] for l in ytick_labels]
ax[0].set_yticklabels(ytick_labels, rotation=90, va='center', ma='center')
for i in range(len(ytick_labels)):
    plt.axhline(y=i + 0.5, xmin=0, xmax=1e4, color='black', linestyle=':', alpha=0.3)


#fig.savefig(plots_path / 'Figure6_mi_rh_f_execution_times.pdf', dpi=300, bbox_inches='tight')

######################################################################################################################

## Figure 13 carbon emission with TS+EL instead of ACM
def best_ML_DL_Raw(results):
    Riemann = []
    Raw = []
    DL = []
    for dataset in results["dataset"].unique():
        results_pipeline = \
            results[results["dataset"] == dataset].groupby(['pipeline', "pipeline_category"], as_index=False)[
                "score"].mean()
        Riemann.append(results_pipeline[results_pipeline["pipeline_category"] == "Riemannian"].sort_values(
            by="score",
            ascending=False
        ).iloc[:1]["pipeline"].values[0])  # remove ACM + TS + SVM from carbon computation
        Raw.append(results_pipeline[results_pipeline["pipeline_category"] == "Raw"].sort_values(
            by="score",
            ascending=False
        ).iloc[:1]["pipeline"].values[0])
        DL.append(results_pipeline[results_pipeline["pipeline_category"] == "Deep Learning"].sort_values(
            by="score",
            ascending=False
        ).iloc[:1]["pipeline"].values[0])

    return max(set(Riemann), key=Riemann.count), max(set(Raw), key=Raw.count), max(set(DL), key=DL.count)


# Best Pipeline
Riemann, Raw, DL = best_ML_DL_Raw(mi_rf_df_withaug)
Riemann = "TS + EL"

df2 = mi_rf_df_withaug[(mi_rf_df_withaug["pipeline"].values == Riemann) |
                       (mi_rf_df_withaug["pipeline"].values == DL) |
                       (mi_rf_df_withaug["pipeline"].values == Raw)]
# df2.pipeline = df2.pipeline.cat.remove_unused_categories()
n_ds = df2.dataset.unique().shape[0]
n_pi = df2.pipeline.unique().shape[0]
n_cat = df2.pipeline_category.unique().shape[0]

# Pipelines (agg across datasets)
sns.boxplot(data=df2,
            y="pipeline",
            x="carbon_emission",
            hue="pipeline_category",
            order=[Riemann, Raw, DL],
            palette=(palettes['pipeline_category'] if n_cat > 1 else "Set2"),
            ax=ax[1],
            dodge=False,
            width=.2,
            linewidth=boxplot_lw,
            boxprops=dict(alpha=0.7),
            orient='h')

handles, labels = ax[1].get_legend_handles_labels()
unique_labels = {}
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels[label] = handle

# ax.set_xlim(0, 1)
ax[1].set_xscale('log')
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ytick_labels = [l.get_text() for l in ax[1].get_yticklabels()]
ytick_labels = [pip_labels_map[l] for l in ytick_labels]

ax[1].set_yticks([x - 0.2 for x in ax[1].get_yticks()])
ax[1].set_yticklabels(ytick_labels, rotation=90, va='center', ma='center')
for i in range(len(ytick_labels)):
    plt.axhline(y=i + 0.3, xmin=0, xmax=1.02, color='black', linestyle=':', alpha=0.3)
ax[1].legend(unique_labels.values(), unique_labels.keys(), loc=(-.5, 1.01), ncols=3, title='Pipeline category')
ax[1].set(xlabel="CO2 Emission (g)\nMI RH-F datasets", ylabel="MI RH-F pipelines")
plt.savefig(plots_path / 'Figure8_comput_time_carbon_emission_2.pdf', bbox_inches='tight')
plt.savefig(plots_path / 'Figure8_comput_time_carbon_emission_2.jpg', bbox_inches='tight')
plt.show()
######################################################################################################################
######################################################################################################################
# # Average raincloudplot (one point per dataset) with the three paradigms (Figure14)

# paradigms = ['mi_lhrh', 'ssvep', 'p300'] # 'mi_all', 'mi_rf'
# f, axs = plt.subplots(1, len(paradigms), figsize=(5*len(paradigms), 2))
# first = True
# for paradigm_name, ax in zip(paradigms, axs):
#     # if paradigm_name=="ssvep":
#     #     y="ITR"
#     # else:
#     #     y='score'
#     y='score'
#     df = dfs[paradigm_name]
#     avg_df = df.groupby(['dataset', 'pipeline_category'])[[y]].mean().reset_index(drop=False)
#     score_name = 'Accuracy score' if paradigm_name in ['mi_all', 'ssvep'] else 'ROC-AUC score'
#     if paradigm_name=='ssvep':
#         score_name="ITR score"
#     avg_df['Paradigm'] = paradigms_map[paradigm_name]
#     # Pipelines (agg across datasets)
    

#     pt.RainCloud(
#         data=avg_df, y=y,x='Paradigm',
#         hue='pipeline_category',
#         bw='scott',
#         width_viol=0.7, width_box=0.3, point_size=5,
#         dodge=True, orient='h',
#         linewidth=0, box_linewidth=boxplot_lw,
#         box_whiskerprops=boxplot_props,
#         box_medianprops=boxplot_props,
#         alpha=0.7, palette=palettes['pipeline_category'],
#         box_showfliers=False,
#         ax=ax, pointplot=True,
#         point_linestyles="none", linecolor='k',
#         point_markers='D',
#         # order=[Riemann, Raw, DL]
#     )
#     if first:
#         handles, labels = ax.get_legend_handles_labels()
#         unique_labels = {}
#         for handle, label in zip(handles, labels):
#             if label not in unique_labels:
#                 unique_labels[label] = handle
#         ax.legend(unique_labels.values(), unique_labels.keys(), loc=(0.0, 1.2), ncols=3,
#                   title='Pipeline category')
#     else:
#         ax.get_legend().remove()
#     ax.set(xlabel=score_name, ylabel=None, yticks=[], yticklabels=[])

#     ax.title.set_text(paradigms_map[paradigm_name])
#     first = False
# plt.savefig(plots_path / 'Figure14_avg_paradigms_accuracy.pdf', bbox_inches='tight')
######################################################################################################################

class MyClass():
    pass

def my_function(first_parameter, second_parameter):
    pass