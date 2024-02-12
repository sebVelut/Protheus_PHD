from pathlib import Path
import json

import pandas as pd

import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns

root = Path(__file__).parent.parent
data_path = root / 'moabb_results'
plots_path = root / 'pictures' / 'moabb_results' / 'rainclouds'
# csvs = data_path.glob('*.csv')

p300_df = pd.read_csv(data_path / 'results_P300.csv')
ssvep_df = pd.read_csv(data_path / 'results_SSVEP.csv')
mi_rf_df = pd.read_csv(data_path / 'results_rf.csv')
mi_lhrh_df = pd.read_csv(data_path / 'results_lhrh.csv')
mi_all_df = pd.read_csv(data_path/'results_all.csv')
palettes = json.load(open(root / 'Scripts' / 'colours.json', 'r'))

dfs = dict(
    mi_all=mi_all_df,
    mi_rf=mi_rf_df,
    mi_lhrh=mi_lhrh_df,
    p300=p300_df,
    ssvep=ssvep_df,
)

mi_pipelines_order = pd.concat([df for n, df in dfs.items() if 'mi' in n]
                               ).groupby(['pipeline','pipeline_category']
                                         ).score.mean().sort_values(ascending=False).reset_index()

for name, df in dfs.items():
    # df['pipeline_category'] = df['pipeline_category'].astype('category')
    dataset_order = df.groupby('dataset').score.mean().sort_values(ascending=False)
    df.dataset = df.dataset.astype(pd.CategoricalDtype(categories=dataset_order.index.to_list()))
    if 'mi' in name:
        df.pipeline = df.pipeline.astype(pd.CategoricalDtype(categories=mi_pipelines_order.pipeline.to_list()))
    else:
        pipelines_order = df.groupby('pipeline').score.mean().sort_values(ascending=False)
        df.pipeline = df.pipeline.astype(pd.CategoricalDtype(categories=pipelines_order.index.to_list()))


def plot_mi(df, name):
    for cat in df['pipeline_category'].unique():
        df2 = df[df['pipeline_category'] == cat].copy()
        df2.pipeline = df2.pipeline.cat.remove_unused_categories()
        full_name = f'{name} - {cat}'
        full_id = f'{name}_{cat.lower().replace(" ", "-")}'
        plot_other(df2, full_name, full_id)

    # best n
    for best_n in [1,3]:
        full_name = f'{name} - best {best_n}'
        full_id = f'{name}_best{best_n}'
        best_pipes = mi_pipelines_order.groupby('pipeline_category').head(best_n)
        df2 = df.copy()[df.pipeline.isin(best_pipes.pipeline.to_list())]
        df2.pipeline = df2.pipeline.cat.remove_unused_categories()
        plot_other(df2, full_name, full_id)

    n_ds = df.dataset.unique().shape[0]
    f, ax = plt.subplots(figsize=(6, n_ds * 1.5))
    # Add average
    avg = df.groupby(['dataset', 'pipeline_category'])[['score']].mean().reset_index(drop=False)
    avg['dataset'] = 'Average'
    df_avg = pd.concat([df, avg])
    # Categories x Datasets
    pt.RainCloud(
        data=df_avg, hue='pipeline_category', y='score', x='dataset',
        bw='scott',
        width_viol=.7, width_box=.5,
        dodge=True, orient='h',
        linewidth=0, box_linewidth=0,
        alpha=0.7, palette=palettes['pipeline_category'],
        box_showfliers=False,
        ax=ax, pointplot=True,
        point_markerfacecolor='k', point_markers='D',
    )
    # ax.set_xlim(0, 1)
    plt.title(name)
    # plt.show()
    plt.savefig(plots_path / f'{name}.pdf', bbox_inches='tight')


def plot_other(df, name, identifier):
    n_ds = df.dataset.unique().shape[0]
    n_pi = df.pipeline.unique().shape[0]
    n_cat = df.pipeline_category.unique().shape[0]
    f, ax = plt.subplots(figsize=(6, n_pi))
    # Pipelines (agg across datasets)
    pt.RainCloud(
        data=df, y='score', x='pipeline',
        hue=('pipeline_category' if n_cat>1 else None),
        bw='scott',
        width_viol=.7, width_box=0.5,
        dodge=False, point_dodge=False, orient='h',
        linewidth=0, box_linewidth=0,
        alpha=0.7, palette=(palettes['pipeline_category'] if n_cat>1 else "Set2"),
        box_showfliers=False,
        ax=ax, pointplot=False,
        point_linestyles="none", linecolor='k',
        point_markers='D',
    )
    # ax.set_xlim(0, 1)
    plt.title(f"{name} - aggregated")
    plt.savefig(plots_path / f'{identifier}_agg.pdf', bbox_inches='tight')


    # Add avereage
    avg = df.groupby(['dataset', 'pipeline','pipeline_category'], observed=True)[['score']].mean().reset_index(drop=False)
    avg['dataset'] = 'Average'
    df_avg = pd.concat([df, avg])
    if n_cat>1:
        df_avg.pipeline = df_avg.apply(lambda row: f"{row.pipeline} ({row.pipeline_category})", axis=1)
    palette = "Set2"
    if n_cat==n_pi:
        # Order categories
        cat_pip_map = df_avg.groupby('pipeline_category')['pipeline'].first().to_dict()
        palette = {cat_pip_map[k]: v for k,v in palettes['pipeline_category'].items()}
        # df_avg.pipeline = df_avg.pipeline.astype(
        #     pd.CategoricalDtype(categories=[cat_pip_map[c] for c in df_avg.pipeline_category.dtype.categories]))
    f, ax = plt.subplots(figsize=(6, n_ds * 1.5))
    # Pipelines x Datasets
    g = pt.RainCloud(
        data=df_avg, hue='pipeline', y='score', x='dataset',
        bw='scott',
        width_viol=.7, width_box=0.5,
        dodge=True, orient='h',
        linewidth=0, box_linewidth=0,
        alpha=0.7, palette=palette,
        box_showfliers=False,
        ax=ax, pointplot=True,
        point_markerfacecolor='k', point_markers='D',
    )
    # ax.set_xlim(0, 1)
    plt.title(name)
    # plt.show()
    plt.savefig(plots_path / f'{identifier}.pdf', bbox_inches='tight')

sns.set_style('whitegrid')
for name, df in dfs.items():
    if 'mi' in name:
        plot_mi(df, name)
    else:
        plot_other(df, name, name)
    # plt.close()
    # plt.show()
