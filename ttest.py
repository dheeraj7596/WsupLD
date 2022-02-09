import pickle
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
from test_significance import main_2

if __name__ == "__main__":
    dataset = "nyt-fine"
    data_path = "./data/" + dataset + "/"
    cls = "bert"

    nofilter_preds = pickle.load(open(data_path + cls + "_pred_labels_0.pkl", "rb"))
    lops_preds = pickle.load(open(data_path + cls + "_pred_labels_1.pkl", "rb"))
    prob_preds = pickle.load(open(data_path + cls + "_pred_labels_2.pkl", "rb"))
    random_preds = pickle.load(open(data_path + cls + "_pred_labels_4.pkl", "rb"))
    stability_preds = pickle.load(open(data_path + cls + "_pred_labels_5.pkl", "rb"))
    true_df = pickle.load(open(data_path + "df.pkl", "rb"))
    true_labels = list(true_df["label"])

    nofilter_df = pd.DataFrame.from_dict({"true": true_labels, "clf1": nofilter_preds, "clf2": lops_preds})
    prob_df = pd.DataFrame.from_dict({"true": true_labels, "clf1": prob_preds, "clf2": lops_preds})
    random_df = pd.DataFrame.from_dict({"true": true_labels, "clf1": random_preds, "clf2": lops_preds})
    stability_df = pd.DataFrame.from_dict({"true": true_labels, "clf1": stability_preds, "clf2": lops_preds})

    dfs = [nofilter_df, random_df, prob_df, stability_df]

    for df in dfs:
        perf1 = []
        perf2 = []
        for i in range(100):
            temp_df = df.sample(frac=0.3).reset_index(drop=True)
            perf1.append(f1_score(temp_df["true"], temp_df["clf1"], average='macro'))
            perf2.append(f1_score(temp_df["true"], temp_df["clf2"], average='macro'))
        main_2(perf1, perf2)
