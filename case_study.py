import pickle
import numpy as np

if __name__ == "__main__":
    data_path = "./data/nyt-fine/"
    correct = pickle.load(open(data_path + "correct_bs.pkl", "rb"))
    wrong = pickle.load(open(data_path + "wrong_bs.pkl", "rb"))

    # index_to_label = {0: 'movies', 1: 'gun_control', 2: 'energy_companies', 3: 'golf', 4: 'television',
    #                   5: 'federal_budget', 6: 'the_affordable_care_act', 7: 'tennis', 8: 'immigration', 9: 'soccer',
    #                   10: 'football', 11: 'music', 12: 'surveillance', 13: 'baseball', 14: 'environment', 15: 'cosmos',
    #                   16: 'gay_rights', 17: 'stocks_and_bonds', 18: 'law_enforcement', 19: 'economy', 20: 'hockey',
    #                   21: 'basketball', 22: 'international_business', 23: 'military', 24: 'abortion', 25: 'dance'}

    index_to_label = {0: 'international_business', 1: 'tennis', 2: 'movies', 3: 'economy', 4: 'dance', 5: 'gun_control',
                      6: 'federal_budget', 7: 'immigration', 8: 'cosmos', 9: 'abortion', 10: 'soccer',
                      11: 'stocks_and_bonds', 12: 'television', 13: 'energy_companies', 14: 'surveillance',
                      15: 'baseball', 16: 'environment', 17: 'law_enforcement', 18: 'the_affordable_care_act',
                      19: 'hockey', 20: 'basketball', 21: 'music', 22: 'football', 23: 'military', 24: 'golf',
                      25: 'gay_rights'}

    top_50_correct = {"text": [], "pred": [], "true": [], "ind": [], "prob": []}
    count = 0
    length = len(correct["text"])
    probs = np.array(correct["prob"] + wrong["prob"])
    for i in range(length):
        if correct["first_ep"][i] == 1:
            count += 1
            # top_50_correct["text"].append(correct["text"][i])
            # top_50_correct["pred"].append(correct["pred"][i])
            # top_50_correct["true"].append(correct["true"][i])
            # top_50_correct["correct"].append(1)

    length = len(wrong["text"])
    for i in range(length):
        if wrong["first_ep"][i] == 1:
            count += 1
            top_50_correct["text"].append(wrong["text"][i])
            top_50_correct["pred"].append(wrong["pred"][i])
            top_50_correct["true"].append(wrong["true"][i])
            top_50_correct["prob"].append(wrong["prob"][i])
            top_50_correct["ind"].append(i)

    print(count)
    temp = np.argsort(probs)[::-1][:count]
    for i in temp:
        if i > len(correct["text"]):
            i = i - len(correct["text"]) - 1
            if i not in top_50_correct["ind"]:
                if wrong["prob"][i] >= 0.895:
                    print(wrong["prob"][i], wrong["first_ep"][i], index_to_label[wrong["pred"][i]], wrong["text"][i],
                          index_to_label[wrong["true"][i]])
                    # print(wrong["prob"][i], wrong["first_ep"][i])
                    print("*" * 80)
    # prob_selected = {}
    #
    # texts = []
    # preds = []
    # true = []
    # first_eps = []
    # probs = []
    #
    # for w in wrong:
    #     if wrong["prob"] > 0.95:
    #
    pass
