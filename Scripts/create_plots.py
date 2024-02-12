import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_score_code(path):
    df = pd.read_csv(path)
    # print(df["score"])
    plt.figure(1)
    sns.lineplot(data=df,x="nb_cal",y="Score",hue="clf",legend="full")
    plt.figure(2)
    sns.lineplot(data=df,x="nb_cal",y="Score",hue="subject",legend="full")
    plt.figure(3)
    sns.lineplot(data=df,x="nb_cal",y="time",hue="clf",legend="full")
    plt.figure(4)
    sns.lineplot(data=df,x="nb_cal",y="time_training",hue="clf",legend="full")
    plt.figure(5)
    sns.lineplot(data=df[df["clf"]=="Xdawn+LDA"],x="nb_cal",y="Score",hue="subject",legend="full")
    plt.plot(np.ones(12)*0.9)
    # plt.figure(10)
    # sns.lineplot(data=df,x="nb_cal",y="Score",hue="subject",legend="full",units="subject")
    plt.show()


plot_score_code("C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\results\\results_avecBalancing.csv")
# plot_score_code("C:\\Users\\s.velut\\Documents\\These\\Protheus_PHD\\results\\results_sansBalancing.csv")
    