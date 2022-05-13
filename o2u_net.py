from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler
import torch
import time
import datetime
from util import *
from torch.nn import CrossEntropyLoss
import os
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_num(dataset, iteration):
    dic = roberta_data_it_dict()
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
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",  # Use the 12-layer BERT model, with an uncased vocab.
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
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', do_lower_case=True)
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
    print("Finally", sample_to_loss, flush=True)
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
