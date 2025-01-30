import pandas as pd
import numpy as np
import sys


# def extract_means(df):
#     for col in df.columns[1:]:
#         df[col] = df[col].str.split("±").str[0].astype(float) / 100
#     return df

def extract_means(df):
    df_temp = df.groupby(['pipeline','dataset'])['score'].mean().reset_index()
    
    return df_temp.pivot(index='pipeline', columns='dataset')

def calculate_ITR(df, num_classes, T_dataset):
    result_df = df.copy()

    # Calculate ITR for each dataset (column)
    for col, M, T in zip(df.columns, num_classes, T_dataset):
        P = df[col]
        if M > 1:
            term1 = np.log2(M)
            term2 = P * np.log2(np.maximum(P, 1e-10))
            term3 = (1 - P) * np.log2(np.maximum((1 - P) / (M - 1), 1e-10))  # Same here
            ITR = (60 / T) * (term1 + term2 + term3)
        else:
            ITR = np.zeros_like(P)

        result_df[col] = ITR
    return result_df

def calculate_ITR_all(df, num_classes, T_dataset):
    result_df = df.copy()
    itr_list = []

    # Calculate ITR for each dataset (column)
    for ind in df.index:
        P = df.loc[ind]['score']
        M = num_classes[df.loc[ind]['dataset']]
        T = T_dataset[df.loc[ind]['dataset']]
        if M > 1:
            term1 = np.log2(M)
            term2 = P * np.log2(np.maximum(P, 1e-10))
            term3 = (1 - P) * np.log2(np.maximum((1 - P) / (M - 1), 1e-10))  # Same here
            ITR = (60 / T) * (term1 + term2 + term3)
        else:
            ITR = np.zeros_like(P)

        itr_list.append(ITR)
    result_df["ITR"] = itr_list

    return result_df



def main(file, flag):
    df = pd.read_csv(file,sep=";")
    # df_cleaned = extract_means(df)
    # print(df_cleaned.head)
    if flag == "SSVEP":
        # SSVEP
        num_classes = {"Kalunga2016":4,"Lee2019-SSVEP":4,"MAMEM1":5,"MAMEM2":5,"MAMEM3":4,"Nakanishi2015":12,"Wang2016":40}
        T_dataset = {"Kalunga2016":2,"Lee2019-SSVEP":1,"MAMEM1":3,"MAMEM2":3,"MAMEM3":3,"Nakanishi2015":4.15,"Wang2016":5}
        # num_classes = [4, 4, 5, 5, 4, 12, 40]
        # T_dataset = [2, 1, 3, 3, 3, 4.15, 5]
    elif flag == "MI":
        # MI
        num_classes = [3, 4, 2, 2, 5, 5, 4, 7, 3]
        T_dataset = [3, 4, 5, 5, 7, 3, 4, 4, 5]
    elif flag == "MI_ALL":
        num_classes = [3, 4, 5, 4, 7, 3]
        T_dataset = [3, 4, 3, 4, 4, 5]
    # Calculate ITR
    itr_df = calculate_ITR_all(df.iloc[:, :], num_classes, T_dataset)
    print(itr_df)
    # itr_df.insert(0, "Pipeline", df_cleaned["pipeline"])

    itr_df.to_csv(f"results_ITR_{flag}_calculated.csv", index=False)
    print(f"ITR calculated and saved to 'ITR_{flag}_calculated.csv'.")

main("C:/Users/s.velut/Documents/These/Protheus_PHD/MOABB/article ITR/results_SSVEP_all.csv","SSVEP")


# if __name__ == "__main__":
#     main(sys.argv[1], str(sys.argv[2]))
