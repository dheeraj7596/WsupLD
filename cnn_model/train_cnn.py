import os
import sys
import torch.autograd as autograd
import torch.nn.functional as F
import torch
import torchtext.legacy.data as data
from sklearn.model_selection import train_test_split
from cnn_model.model import CNN_Text
from cnn_model.dataset import TrainValFullDataset
from collections import Counter
from util import compute_train_non_train_inds
import numpy as np
import time


def filter(X_train, y_train, y_true, percent_thresh, device, text_field, label_field, it):
    torch.cuda.empty_cache()
    inds_map = {}
    for i, j in enumerate(y_train):
        try:
            inds_map[j].append(i)
        except:
            inds_map[j] = [i]

    thresh_map = dict(Counter(y_train))
    print("Counts of pseudo-labels ", thresh_map, flush=True)
    for i in thresh_map:
        thresh_map[i] = int(thresh_map[i] * percent_thresh)
    print("Threshold map ", thresh_map, flush=True)

    train_X, val_X, train_y, val_y = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
    train_data, val_data, full_data = TrainValFullDataset.splits(text_field, label_field, train_X, train_y, val_X,
                                                                 val_y, X_train, y_true)

    train_iter, dev_iter, full_data_iter = data.BucketIterator.splits((train_data, val_data, full_data),
                                                                      batch_sizes=(128, 128, 16), sort=False,
                                                                      sort_within_batch=False)

    embed_num = len(text_field.vocab)
    class_num = len(label_field.vocab)
    kernel_sizes = [3, 4, 5]
    cnn = CNN_Text(
        embed_num=embed_num,
        class_num=class_num,
        kernel_sizes=kernel_sizes)
    if device is not None:
        cnn = cnn.to(device)

    model = cnn
    lr = 0.001
    num_epochs = 256
    early_stop = 3
    log_interval = 100

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    filter_flag_map = {}
    train_inds_map = {}
    non_train_inds_map = {}

    for i in thresh_map:
        filter_flag_map[i] = False
        train_inds_map[i] = set([])
        non_train_inds_map[i] = set([])

    steps = 0
    best_epoch = 0
    best_model = None
    best_loss = float("inf")
    stop_flag = False

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature.t_()  # batch first
            if device is not None:
                feature, target = feature.to(device), target.to(device)

            optimizer.zero_grad()
            logit = model(feature)
            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()) == target).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))

        torch.cuda.empty_cache()
        pred_labels, pred_probs, true_labels = test_eval(full_data_iter, model, device)

        count = 0
        for lbl in filter_flag_map:
            if not filter_flag_map[lbl]:
                train_inds, non_train_inds = compute_train_non_train_inds(pred_labels, y_train, inds_map, lbl)
                train_inds_map[lbl].update(set(train_inds))
                non_train_inds_map[lbl].update(set(non_train_inds))

                if len(train_inds_map[lbl]) >= thresh_map[lbl]:
                    filter_flag_map[lbl] = True
                    count += 1
            else:
                count += 1

        print("Number of labels reached 50 percent threshold", count, flush=True)
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                print("For label ", i, " Number expected ", thresh_map[i], " Found ", len(train_inds_map[i]),
                      flush=True)

        temp_flg = True
        for i in filter_flag_map:
            temp_flg = temp_flg and filter_flag_map[i]

        if temp_flg:
            stop_flag = True
            break

        dev_loss = eval(dev_iter, model, device)
        if dev_loss <= best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            best_model = model
        else:
            if epoch - best_epoch >= early_stop:
                print('early stop by {} epochs.'.format(early_stop), flush=True)
                print("Best epoch: ", best_epoch, "Current epoch: ", epoch, flush=True)
                break

    if not stop_flag:
        print("MAX EPOCHS REACHED!!!!!!", flush=True)
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                print("Resetting train, non-train inds for label ", i)
                train_inds_map[i] = inds_map[i]
                non_train_inds_map[i] = []

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    for lbl in train_inds_map:
        for loop_ind in train_inds_map[lbl]:
            train_data.append(X_train[loop_ind])
            train_labels.append(y_train[loop_ind])
            true_train_labels.append(y_true[loop_ind])

    for lbl in non_train_inds_map:
        for loop_ind in non_train_inds_map[lbl]:
            non_train_data.append(X_train[loop_ind])
            non_train_labels.append(y_train[loop_ind])
            true_non_train_labels.append(y_true[loop_ind])

    torch.cuda.empty_cache()
    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def train(train_iter, dev_iter, text_field, label_field, model, device, correct, wrong, lr, num_epochs, early_stop=5,
          log_interval=100, label_dyn=False):
    if device is not None:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    steps = 0
    best_epoch = 0
    best_model = None
    best_loss = float("inf")
    epochs_run = 0
    for epoch in range(1, num_epochs + 1):
        print("Epoch:", epoch, flush=True)
        model.train()
        epochs_run += 1
        for batch in train_iter:
            start_batch = time.time()
            start_t = time.time()
            feature, target = batch.text, batch.label
            print("Getting feature, target from batch", time.time() - start_t, flush=True)
            start_t = time.time()
            feature.t_()  # batch first
            print("Making batch first", time.time() - start_t, flush=True)
            start_t = time.time()
            if device is not None:
                feature, target = feature.to(device), target.to(device)
            print("Moving to device", time.time() - start_t, flush=True)

            start_t = time.time()
            optimizer.zero_grad()
            print("zero grad", time.time() - start_t, flush=True)
            start_t = time.time()
            logit = model(feature)
            print("Model", time.time() - start_t, flush=True)
            start_t = time.time()
            loss = F.cross_entropy(logit, target)
            print("Loss comp", time.time() - start_t, flush=True)
            start_t = time.time()
            loss.backward()
            print("Loss backward", time.time() - start_t, flush=True)
            start_t = time.time()
            optimizer.step()
            print("Opt step", time.time() - start_t, flush=True)

            steps += 1
            if steps % log_interval == 0:
                start_t = time.time()
                corrects = (torch.max(logit, 1)[1].view(target.size()) == target).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy.item(),
                                                                             corrects.item(),
                                                                             batch.batch_size))
                print("Accuracy", time.time() - start_t, flush=True)

            print("Batch processing time", time.time() - start_batch, flush=True)
            print("*" * 80)

        torch.cuda.empty_cache()

        if label_dyn:
            if len(correct["text"]) > 0:
                cor_bs_label_inds, _ = test(model, correct["text"], correct["pred"], text_field, label_field, device)
                for index, pred_ind in enumerate(cor_bs_label_inds):
                    if pred_ind == correct["pred"][index]:
                        correct["match"][index] += 1
                        if correct["first_ep"][index] == 0:
                            correct["first_ep"][index] = epoch

            if len(wrong["text"]) > 0:
                wrong_bs_label_inds, _ = test(model, wrong["text"], wrong["pred"], text_field, label_field, device)
                for index, pred_ind in enumerate(wrong_bs_label_inds):
                    if pred_ind == wrong["pred"][index]:
                        wrong["match"][index] += 1
                        if wrong["first_ep"][index] == 0:
                            wrong["first_ep"][index] = epoch

        start_t = time.time()
        dev_loss = eval(dev_iter, model, device)
        print("Evaluation on val time", time.time() - start_t)
        if dev_loss <= best_loss:
            best_loss = dev_loss
            best_epoch = epoch
            best_model = model
        else:
            if epoch - best_epoch >= early_stop:
                print('early stop by {} epochs.'.format(early_stop), flush=True)
                print("Best epoch: ", best_epoch, "Current epoch: ", epoch, flush=True)
                break

    if label_dyn:
        correct["match"] = list(np.array(correct["match"]) / epochs_run)
        wrong["match"] = list(np.array(wrong["match"]) / epochs_run)

    return best_model, correct, wrong


def train_cnn(X, y, device, text_field, label_field, correct_bootstrap, wrong_bootstrap, label_dyn=False):
    start_t = time.time()
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.1)
    train_data, val_data = TrainValFullDataset.splits(text_field, label_field, train_X, train_y, val_X,
                                                      val_y, None, None)
    print("For generating train_data, val_data", time.time() - start_t, flush=True)
    start_t = time.time()
    train_iter, dev_iter = data.BucketIterator.splits((train_data, val_data), batch_sizes=(128, 128),
                                                      sort=False, sort_within_batch=False)
    print("For generating train_iter, dev_iter", time.time() - start_t, flush=True)

    embed_num = len(text_field.vocab)
    class_num = len(label_field.vocab)
    kernel_sizes = [3, 4, 5]
    cnn = CNN_Text(
        embed_num=embed_num,
        class_num=class_num,
        kernel_sizes=kernel_sizes)
    if device is not None:
        cnn = cnn.to(device)
    model, correct_bootstrap, wrong_bootstrap = train(train_iter, dev_iter, text_field, label_field, cnn, device,
                                                      correct_bootstrap, wrong_bootstrap,
                                                      lr=0.001, num_epochs=256, label_dyn=label_dyn)
    return model, correct_bootstrap, wrong_bootstrap


def eval(data_iter, model, device):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_()  # batch first
        if device is not None:
            feature, target = feature.to(device), target.to(device)

        with torch.no_grad():
            logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)[1].view(target.size()) == target).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size), flush=True)
    torch.cuda.empty_cache()
    return avg_loss


def test_eval(data_iter, model, device):
    model.eval()
    pred_labels = []
    true_labels = []
    total_probs = []
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.t_()  # batch first
        if device is not None:
            feature, target = feature.to(device), target.to(device)

        with torch.no_grad():
            logit = model(feature)

        probs = F.softmax(logit, dim=-1)
        pred_labels.append(torch.max(logit, 1)[1].view(target.size()))
        total_probs.append(probs)
        true_labels.append(target)

    pred_probs = torch.cat(total_probs).contiguous().detach().cpu().numpy()
    pred_labels = torch.cat(pred_labels).contiguous().detach().cpu().numpy()
    true_labels = torch.cat(true_labels).contiguous().detach().cpu().numpy()

    torch.cuda.empty_cache()
    return pred_labels, pred_probs, true_labels


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.item() + 1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)


def test(model, X, y, text_field, label_field, device):
    dataset = TrainValFullDataset(X, y, text_field, label_field)
    iterator = data.Iterator(dataset, batch_size=16, train=False, shuffle=False, repeat=False, sort=False,
                             sort_within_batch=False)

    pred_labels, pred_probs, true_labels = test_eval(iterator, model, device)
    return pred_labels, pred_probs


if __name__ == "__main__":
    X = ["As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love",
         "As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love",
         "As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love",
         "As kids, we lived together",
         "we fnust20, we laughed, we cried",
         "we did not always show the love"
         ]
    y = [0, 2, 0, 4, 5, 6, 0, 4, 5, 3, 2, 1]
    X_full = ["As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love",
              "As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love",
              "As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love",
              "As kids, we lived together",
              "we fnust20, we laughed, we cried",
              "we did not always show the love"
              ]
    y_full = [0, 2, 0, 4, 5, 6, 0, 4, 5, 3, 2, 1]
    use_gpu = False
    save_dir = "./data/"
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False,
                             use_vocab=False,
                             pad_token=None,
                             unk_token=None
                             )

    device = torch.device("cpu")
    text_field.build_vocab(X)
    label_field.build_vocab(y)
    train_cnn(X, y, device, text_field, label_field, None, None, False)
