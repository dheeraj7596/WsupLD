import pickle
import json
from sklearn.metrics import classification_report


def func(df, label_term_dict):
    preds = []
    true = []
    for l in label_term_dict:
        in_filter = "|".join(label_term_dict[l])
        out_filter = ""
        for j in label_term_dict:
            if j == l:
                continue
            if len(out_filter) == 0:
                out_filter += "|".join(label_term_dict[j])
            else:
                out_filter = out_filter + "|" + "|".join(label_term_dict[j])
        pseudo_df = df[df.text.str.contains(in_filter) & ~df.text.str.contains(out_filter)].reset_index(drop=True)
        true += list(pseudo_df["label"])
        preds += ([l] * len(pseudo_df))
    correct_count = 0
    wrong_count = 0
    for i in range(len(preds)):
        if true[i] == preds[i]:
            correct_count += 1
        else:
            wrong_count += 1
    print("Noise Ratio", wrong_count / len(preds))
    print(classification_report(true, preds), flush=True)


if __name__ == "__main__":
    basepath = "/Users/dheerajmekala/Work/WsupLD/data/"
    # basepath = "/data4/dheeraj/coarse2fine/"
    dataset = "nyt-fine/"
    pkl_dump_dir = basepath + dataset

    with open(pkl_dump_dir + "seedwords.json") as fp:
        label_term_dict = json.load(fp)

    df = pickle.load(open(pkl_dump_dir + "df.pkl", "rb"))
    func(df, label_term_dict)
