from transformers import BertForSequenceClassification, BertTokenizerFast, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch
import time
import random
import datetime
from util import *
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter
from torch.nn import CrossEntropyLoss
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_num(dataset, iteration):
    dic = bert_data_it_dict()
    try:
        num = dic[dataset][iteration]
    except:
        if dataset not in dic:
            raise Exception("Dataset out of bounds " + dataset)
        elif iteration not in dic[dataset]:
            raise Exception("Iteration out of bounds " + dataset + str(iteration))
        else:
            raise Exception("Something went wrong " + dataset + str(iteration))
    return num


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def bert_tokenize(tokenizer, sentences, labels):
    temp = tokenizer(
        sentences,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    input_ids = temp["input_ids"]
    attention_masks = temp["attention_mask"]
    labels = torch.tensor(labels)
    # Print sentence 0, now as a list of IDs.
    # print('Original: ', sentences[0])
    # print('Token IDs:', input_ids[0])
    return input_ids, attention_masks, labels


def create_data_loaders(dataset):
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # The DataLoader needs to know our batch size for training, so we specify it
    # here. For fine-tuning BERT on a specific task, the authors recommend a batch
    # size of 16 or 32.
    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size,  # Evaluate with this batch size.
        pin_memory=True,
        num_workers=16
    )
    return train_dataloader, validation_dataloader


def train(train_dataloader, validation_dataloader, device, num_labels, correct, wrong, label_dyn=False):
    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("", flush=True)
        print("Running Validation...", flush=True)

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy), flush=True)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss), flush=True)
        print("  Validation took: {:}".format(validation_time), flush=True)

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if not label_dyn:
            continue

        if len(correct["text"]) > 0:
            corr_bs_predictions = test(model, correct["text"], correct["pred"], device)
            cor_bs_label_inds = get_labelinds_from_probs(corr_bs_predictions)
            for index, pred_ind in enumerate(cor_bs_label_inds):
                if pred_ind == correct["pred"][index]:
                    correct["match"][index] += 1
                    if correct["first_ep"][index] == 0:
                        correct["first_ep"][index] = epoch_i + 1

        if len(wrong["text"]) > 0:
            wrong_bs_predictions = test(model, wrong["text"], wrong["pred"], device)
            wrong_bs_label_inds = get_labelinds_from_probs(wrong_bs_predictions)
            for index, pred_ind in enumerate(wrong_bs_label_inds):
                if pred_ind == wrong["pred"][index]:
                    wrong["match"][index] += 1
                    if wrong["first_ep"][index] == 0:
                        wrong["first_ep"][index] = epoch_i + 1

    if label_dyn:
        correct["match"] = list(np.array(correct["match"]) / epochs)
        wrong["match"] = list(np.array(wrong["match"]) / epochs)
    print("", flush=True)
    print("Training complete!", flush=True)

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)), flush=True)
    return model, correct, wrong


def evaluate(model, prediction_dataloader, device, mc_dropout=False):
    # Put model in evaluation mode
    model.eval()
    if mc_dropout:
        enable_dropout(model)

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs.logits

        # Move logits and labels to CPU
        logits = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    return predictions, true_labels


def test(model, X_test, y_test, device):
    start = time.time()
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X_test, y_test)
    print("Tokenizing text time:", time.time() - start, flush=True)
    batch_size = 32
    # Create the DataLoader.
    start = time.time()
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )
    print("Dataloader creation time:", time.time() - start, flush=True)
    start = time.time()
    predictions, true_labels = evaluate(model, prediction_dataloader, device)
    print("Evaluation time:", time.time() - start, flush=True)
    return predictions


def train_bert(X, y, device, correct, wrong, label_dyn=False):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    train_dataloader, validation_dataloader = create_data_loaders(dataset)

    # Tell pytorch to run this model on the GPU.

    model, correct, wrong = train(train_dataloader,
                                  validation_dataloader,
                                  device,
                                  len(set(y)),
                                  correct,
                                  wrong,
                                  label_dyn)
    return model, correct, wrong


def filter(X, y_pseudo, y_true, device, percent_thresh=0.5, iteration=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))

    stop_flag = False
    inds_map = {}
    for i, j in enumerate(y_pseudo):
        try:
            inds_map[j].append(i)
        except:
            inds_map[j] = [i]

    thresh_map = dict(Counter(y_pseudo))
    print("Counts of pseudo-labels ", thresh_map, flush=True)
    for i in thresh_map:
        thresh_map[i] = int(thresh_map[i] * percent_thresh)

    print("Threshold map ", thresh_map, flush=True)

    filter_flag_map = {}
    train_inds_map = {}
    non_train_inds_map = {}
    for i in thresh_map:
        filter_flag_map[i] = False
        train_inds_map[i] = []
        non_train_inds_map[i] = []
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)
    epoch_i = 0
    while not stop_flag and epoch_i < epochs:

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

        prediction_sampler = SequentialSampler(dataset)
        prediction_dataloader = DataLoader(dataset,
                                           sampler=prediction_sampler,
                                           batch_size=batch_size,
                                           num_workers=16,
                                           pin_memory=True
                                           )

        first_ep_preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device)
        first_ep_pred_inds = get_labelinds_from_probs(first_ep_preds)

        count = 0
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                train_inds, non_train_inds = compute_train_non_train_inds(first_ep_pred_inds, y_pseudo, inds_map, i)
                train_inds_map[i] = train_inds
                non_train_inds_map[i] = non_train_inds
                if len(train_inds) >= thresh_map[i]:
                    filter_flag_map[i] = True
                    count += 1
            else:
                count += 1

        print("Number of labels reached 50 percent threshold", count)
        for i in filter_flag_map:
            if not filter_flag_map[i]:
                print("For label ", i, " Number expected ", thresh_map[i], " Found ", len(train_inds_map[i]))

        temp_flg = True
        for i in filter_flag_map:
            temp_flg = temp_flg and filter_flag_map[i]
        stop_flag = temp_flg
        epoch_i += 1

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
            train_data.append(X[loop_ind])
            train_labels.append(y_pseudo[loop_ind])
            true_train_labels.append(y_true[loop_ind])

    for lbl in non_train_inds_map:
        for loop_ind in non_train_inds_map[lbl]:
            non_train_data.append(X[loop_ind])
            non_train_labels.append(y_pseudo[loop_ind])
            true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def get_true_label_probs(predictions, true):
    for i, p in enumerate(predictions):
        if i == 0:
            pred = p
        else:
            pred = np.concatenate((pred, p))

    true_inds = np.expand_dims(np.array(true), axis=1)
    probs = np.take_along_axis(pred, true_inds, axis=1)
    probs = probs.reshape(probs.shape[0])
    return probs


def prob_num_filter(X, y_pseudo, y_true, device, dataset_name, iteration, thresh=0.9):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    first_ep_preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device)
    probs = get_true_label_probs(first_ep_preds, y_pseudo)
    train_data_inds = list(np.nonzero(probs >= thresh)[0])
    cutoff_prob = thresh

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    non_train_data_inds = list(set(range(len(y_pseudo))) - set(train_data_inds))
    for loop_ind in train_data_inds:
        train_data.append(X[loop_ind])
        train_labels.append(y_pseudo[loop_ind])
        true_train_labels.append(y_true[loop_ind])

    for loop_ind in non_train_data_inds:
        non_train_data.append(X[loop_ind])
        non_train_labels.append(y_pseudo[loop_ind])
        true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels, probs, cutoff_prob


def prob_filter(X, y_pseudo, y_true, device, dataset_name, iteration, num=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    first_ep_preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device)
    probs = get_true_label_probs(first_ep_preds, y_pseudo)
    inds = list(np.argsort(probs)[::-1])
    if num is None:
        num = get_num(dataset_name, iteration)

    train_data_inds = inds[:num]
    length = len(X)
    if num > length:
        cutoff_prob = probs[-1]
    else:
        cutoff_prob = probs[inds[num - 1]]

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    non_train_data_inds = list(set(range(len(y_pseudo))) - set(train_data_inds))
    for loop_ind in train_data_inds:
        train_data.append(X[loop_ind])
        train_labels.append(y_pseudo[loop_ind])
        true_train_labels.append(y_true[loop_ind])

    for loop_ind in non_train_data_inds:
        non_train_data.append(X[loop_ind])
        non_train_labels.append(y_pseudo[loop_ind])
        true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels, probs, cutoff_prob


def mc_dropout_filter(X, y_pseudo, y_true, device, dataset_name, iteration, num=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    variance_list = compute_mc_dropout(prediction_dataloader, y_pseudo, 5, model, device)
    inds = list(np.argsort(variance_list))
    if num is None:
        num = get_num(dataset_name, iteration)

    train_data_inds = inds[:num]
    length = len(X)

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    non_train_data_inds = list(set(range(len(y_pseudo))) - set(train_data_inds))
    for loop_ind in train_data_inds:
        train_data.append(X[loop_ind])
        train_labels.append(y_pseudo[loop_ind])
        true_train_labels.append(y_true[loop_ind])

    for loop_ind in non_train_data_inds:
        non_train_data.append(X[loop_ind])
        non_train_labels.append(y_pseudo[loop_ind])
        true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def mc_dropout_entropy_filter(X, y_pseudo, y_true, device, dataset_name, iteration, num=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    entropy_list = compute_mc_dropout_entropy(prediction_dataloader, y_pseudo, 5, model, device)
    inds = list(np.argsort(entropy_list))
    if num is None:
        num = get_num(dataset_name, iteration)

    train_data_inds = inds[:num]
    length = len(X)

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    non_train_data_inds = list(set(range(len(y_pseudo))) - set(train_data_inds))
    for loop_ind in train_data_inds:
        train_data.append(X[loop_ind])
        train_labels.append(y_pseudo[loop_ind])
        true_train_labels.append(y_true[loop_ind])

    for loop_ind in non_train_data_inds:
        non_train_data.append(X[loop_ind])
        non_train_labels.append(y_pseudo[loop_ind])
        true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()


def compute_mc_dropout(prediction_dataloader,
                       y_pseudo,
                       forward_passes,
                       model,
                       device):
    n_samples = len(y_pseudo)
    true_index_probs = np.empty((n_samples, 0))
    for i in range(forward_passes):
        print("Forward pass counter:", i, flush=True)
        first_ep_preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device, mc_dropout=True)
        probs = get_true_label_probs(first_ep_preds, y_pseudo)
        probs = probs.reshape((probs.shape[0], 1))
        true_index_probs = np.hstack((true_index_probs, probs))
        print("Shape of true_index_probs", true_index_probs.shape, flush=True)

    variance = np.var(true_index_probs, axis=-1)  # shape (n_samples, 1)
    return variance


def compute_mc_dropout_entropy(prediction_dataloader,
                               y_pseudo,
                               forward_passes,
                               model,
                               device):
    n_samples = len(y_pseudo)
    n_classes = len(set(y_pseudo))

    dropout_predictions = np.empty((0, n_samples, n_classes))
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)

        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs.logits
            # Move logits and labels to CPU
            logits = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            # Store predictions and true labels
            predictions = np.vstack((predictions, logits))
            print("Shape of predictions", predictions.shape, flush=True)

        dropout_predictions = np.vstack((dropout_predictions, predictions[np.newaxis, :, :]))
        print("Shape of dropout_predictions", dropout_predictions.shape, flush=True)
        # dropout predictions - shape (forward_passes, n_samples, n_classes)

    # Calculating mean across multiple MCD forward passes
    mean = np.mean(dropout_predictions, axis=0)  # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes
    entropy = -np.sum(mean * np.log(mean + epsilon), axis=-1)  # shape (n_samples,)
    print("Shape of entropy", entropy.shape, flush=True)
    return entropy


def dump_probs(X, y_pseudo, y_true, device, data_path):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    first_ep_preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device)
    probs = get_true_label_probs(first_ep_preds, y_pseudo)

    pickle.dump(probs, open(data_path + "probs_prob_filter.pkl", "wb"))
    pickle.dump(y_pseudo, open(data_path + "y_pseudo_prob_filter.pkl", "wb"))
    pickle.dump(y_true, open(data_path + "y_true_prob_filter.pkl", "wb"))


def prob_score_filter(X, y_pseudo, y_true, device, dataset_name, iteration):
    match = []
    for i in X:
        match.append(0)
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    print("Getting data that can be trained in 1 epoch..", flush=True)

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

        ep_preds, ep_true_labels = evaluate(model, prediction_dataloader, device)
        ep_pred_inds = get_labelinds_from_probs(ep_preds)
        length = len(ep_pred_inds)
        for loop_variable in range(length):
            if ep_pred_inds[loop_variable] == y_pseudo[loop_variable]:
                match[loop_variable] += 1

    inds = list(np.argsort(match)[::-1])
    num = get_num(dataset_name, iteration)

    train_data_inds = inds[:num]

    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    non_train_data_inds = list(set(range(len(y_pseudo))) - set(train_data_inds))
    for loop_ind in train_data_inds:
        train_data.append(X[loop_ind])
        train_labels.append(y_pseudo[loop_ind])
        true_train_labels.append(y_true[loop_ind])

    for loop_ind in non_train_data_inds:
        non_train_data.append(X[loop_ind])
        non_train_labels.append(y_pseudo[loop_ind])
        true_non_train_labels.append(y_true[loop_ind])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def random_filter(X, y_pseudo, y_true, iteration, dataset):
    num = get_num(dataset, iteration)
    length = len(X)
    if num >= length:
        return X, y_pseudo, y_true, [], [], []
    else:
        train_data, non_train_data, train_labels, non_train_labels, true_train_labels, true_non_train_labels = train_test_split(
            X, y_pseudo, y_true, stratify=y_pseudo, test_size=(length - num))
        return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def compute_correct_wrong_coverage(model, prediction_dataloader, device, y_pseudo, y_true, correct_dic):
    preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device)
    pred_inds = get_labelinds_from_probs(preds)

    filtered_labels = []
    filtered_true_labels = []
    for ind in range(len(y_pseudo)):
        if pred_inds[ind] == y_pseudo[ind] or correct_dic[ind]:
            correct_dic[ind] = 1
            filtered_labels.append(y_pseudo[ind])
            filtered_true_labels.append(y_true[ind])

    coverage = len(filtered_labels)
    correct = 0
    wrong = 0
    for i, j in zip(filtered_labels, filtered_true_labels):
        if i == j:
            correct += 1
        else:
            wrong += 1
    return correct, wrong, coverage, correct_dic


def batch_epoch_inds_filter(X, y_pseudo, y_true, device, percent_thresh=0.5, batch_epoch=40, iteration=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))

    stop_flag = False
    inds_map = {}
    for i, j in enumerate(y_pseudo):
        try:
            inds_map[j].append(i)
        except:
            inds_map[j] = [i]

    thresh_map = dict(Counter(y_pseudo))
    print("Counts of pseudo-labels ", thresh_map, flush=True)
    for i in thresh_map:
        thresh_map[i] = int(thresh_map[i] * percent_thresh)
    print("Threshold map ", thresh_map, flush=True)

    filter_flag_map = {}
    train_inds_map = {}
    non_train_inds_map = {}
    for i in thresh_map:
        filter_flag_map[i] = False
        train_inds_map[i] = []
        non_train_inds_map[i] = []
    # Tell pytorch to run this model on the GPU.

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=128,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    for epoch_i in range(epochs):
        if stop_flag:
            break

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % batch_epoch == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)
                first_ep_preds, first_ep_true_labels = evaluate(model, prediction_dataloader, device)
                first_ep_pred_inds = get_labelinds_from_probs(first_ep_preds)
                count = 0
                for i in filter_flag_map:
                    if not filter_flag_map[i]:
                        train_inds, non_train_inds = compute_train_non_train_inds(first_ep_pred_inds, y_pseudo,
                                                                                  inds_map, i)
                        train_inds_map[i] = train_inds
                        non_train_inds_map[i] = non_train_inds
                        if len(train_inds) >= thresh_map[i]:
                            filter_flag_map[i] = True
                            count += 1
                    else:
                        count += 1

                print("Number of labels reached 50 percent threshold", count)
                for i in filter_flag_map:
                    if not filter_flag_map[i]:
                        print("For label ", i, " Number expected ", thresh_map[i], " Found ", len(train_inds_map[i]))

                temp_flg = True
                for i in filter_flag_map:
                    temp_flg = temp_flg and filter_flag_map[i]
                stop_flag = temp_flg

            if stop_flag:
                break
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

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
            train_data.append(X[loop_ind])
            train_labels.append(y_pseudo[loop_ind])
            true_train_labels.append(y_true[loop_ind])
    for lbl in non_train_inds_map:
        for loop_ind in non_train_inds_map[lbl]:
            non_train_data.append(X[loop_ind])
            non_train_labels.append(y_pseudo[loop_ind])
            true_non_train_labels.append(y_true[loop_ind])
    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


def batch_epoch_filter(X, y_pseudo, y_true, device, percent_thresh=0.5, batch_epoch=40, iteration=None):
    correct_list = []
    wrong_list = []
    coverage_list = []

    correct_dic = []
    for i in range(len(X)):
        correct_dic.append(0)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels = bert_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 4

    prediction_sampler = SequentialSampler(dataset)
    prediction_dataloader = DataLoader(dataset,
                                       sampler=prediction_sampler,
                                       batch_size=batch_size,
                                       num_workers=16,
                                       pin_memory=True
                                       )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...

    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % batch_epoch == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)
                correct, wrong, coverage, correct_dic = compute_correct_wrong_coverage(model, prediction_dataloader,
                                                                                       device, y_pseudo, y_true,
                                                                                       correct_dic)
                correct_list.append(correct)
                wrong_list.append(wrong)
                coverage_list.append(coverage)
                print("Coverage:", coverage, flush=True)
                print("Correct:", correct, flush=True)
                print("Wrong:", wrong, flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)
    return correct_list, wrong_list, coverage_list


def o2u_tokenize(tokenizer, sentences, labels):
    temp = tokenizer(
        sentences,
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=512,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
    )
    input_ids = temp["input_ids"]
    attention_masks = temp["attention_mask"]
    labels = torch.tensor(labels)
    inds = torch.tensor(list(range(len(sentences))))
    # Print sentence 0, now as a list of IDs.
    # print('Original: ', sentences[0])
    # print('Token IDs:', input_ids[0])
    return input_ids, attention_masks, labels, inds


def first_stage(train_dataloader, device, num_labels):
    print("First stage started", flush=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=num_labels,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )
    model = model.to(device)
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 1
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    for epoch_i in range(epochs):
        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        t0 = time.time()
        total_train_loss = 0

        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)
    return model


def second_stage(model, train_dataloader, device, num_labels):
    print("Second stage started", flush=True)
    sample_to_loss = {}

    model = model.to(device)
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    # Number of training epochs. The BERT authors recommend between 2 and 4.
    # We chose to run for 4, but we'll see later that this may be over-fitting the
    # training data.
    epochs = 3
    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128
    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    # For each epoch...
    for epoch_i in range(epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("", flush=True)
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs), flush=True)
        print('Training...', flush=True)

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()
        data_time = AverageMeter('Data loading time', ':6.3f')
        batch_time = AverageMeter('Batch processing time', ':6.3f')
        # For each batch of training data...
        end = time.time()
        for step, batch in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed),
                      flush=True)

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            b_inds = batch[3].detach().cpu().numpy()

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs.loss
            logits = outputs.logits
            loss_fct = CrossEntropyLoss(reduction='none')
            loss_each = loss_fct(logits.view(-1, num_labels), b_labels.view(-1)).detach().cpu().numpy()
            # print("Shape of loss_each", loss_each.shape, flush=True)
            for loop_ind, ind in enumerate(b_inds):
                try:
                    sample_to_loss[ind].append(loss_each[loop_ind])
                except:
                    sample_to_loss[ind] = [loss_each[loop_ind]]
            # print("Adding to dict", sample_to_loss, flush=True)
            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

        print(str(data_time), flush=True)
        print(str(batch_time), flush=True)
        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Training epoch took: {:}".format(training_time), flush=True)

    for i in sample_to_loss:
        sample_to_loss[i] = np.mean(sample_to_loss[i])

    sample_to_loss = {k: v for k, v in sorted(sample_to_loss.items(), key=lambda item: item[1])}
    return sample_to_loss


def o2u(X, y_pseudo, y_true, device, dataset_name, iteration=None):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', do_lower_case=True)
    start = time.time()
    input_ids, attention_masks, labels, inds = o2u_tokenize(tokenizer, X, y_pseudo)
    print("Time taken in tokenizing:", time.time() - start)

    # Combine the training inputs into a TensorDataset.
    dataset = TensorDataset(input_ids, attention_masks, labels, inds)

    batch_size = 32
    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    start = time.time()
    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=batch_size,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    print("Time taken in initializing dataloader:", time.time() - start)

    num_labels = len(set(y_pseudo))
    model = first_stage(train_dataloader, device, num_labels)

    train_dataloader = DataLoader(
        dataset,  # The training samples.
        sampler=RandomSampler(dataset),  # Select batches randomly
        batch_size=16,  # Trains with this batch size.
        pin_memory=True,
        num_workers=16
    )
    sample_to_loss = second_stage(model, train_dataloader, device, num_labels)
    # print("Finally", sample_to_loss, flush=True)
    num = get_num(dataset_name, iteration)
    selected_inds = []
    count = 0
    for i in sample_to_loss:
        if count >= num:
            break
        selected_inds.append(i)
        count += 1

    non_selected_inds = list(set(range(len(X))) - set(selected_inds))
    train_data = []
    train_labels = []
    true_train_labels = []
    non_train_data = []
    non_train_labels = []
    true_non_train_labels = []

    for i in selected_inds:
        train_data.append(X[i])
        train_labels.append(y_pseudo[i])
        true_train_labels.append(y_true[i])

    for i in non_selected_inds:
        non_train_data.append(X[i])
        non_train_labels.append(y_pseudo[i])
        true_non_train_labels.append(y_true[i])

    return train_data, train_labels, true_train_labels, non_train_data, non_train_labels, true_non_train_labels


if __name__ == "__main__":
    # base_path = "./data/"
    base_path = "/data/dheeraj/WsupLD/data/"
    dataset = sys.argv[3]
    data_path = base_path + dataset + "/"
    use_gpu = int(sys.argv[1])
    gpu_id = int(sys.argv[2])
    # use_gpu = False

    # Tell pytorch to run this model on the GPU.
    if use_gpu:
        device = torch.device('cuda:' + str(gpu_id))
    else:
        device = torch.device("cpu")

    df = pickle.load(open(data_path + "df.pkl", "rb"))

    labels = list(set(df["label"]))
    label_to_index, index_to_label = create_label_index_maps(labels)

    X_all = list(df["text"])
    y_all = list(df["label"])
    y_all_inds = [label_to_index[l] for l in y_all]

    X_train, X_test, y_train, y_test, y_train_inds, y_test_inds = train_test_split(X_all, y_all, y_all_inds,
                                                                                   stratify=y_all, test_size=0.1)
    # Tokenize all of the sentences and map the tokens to their word IDs.
    model, _, _ = train_bert(X_train, y_train_inds, device, None, None, label_dyn=False)

    predictions = test(model, X_test, y_test_inds, device)
    pred_inds = get_labelinds_from_probs(predictions)
    pred_labels = []
    for p in pred_inds:
        pred_labels.append(index_to_label[p])
    print(classification_report(y_test, pred_labels), flush=True)
    print("*" * 80, flush=True)
