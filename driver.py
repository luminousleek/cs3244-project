import copy
import time
from os.path import exists

from torch import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

import torchmetrics

from dataset import JobPostingDataSet
from model import collate_batch, dataset as ds, device, TextClassificationModel, save_model, load_model

# at beginning of the script

def train(dataloader, model, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count), elapsed)
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            # print("Predicted label: ", predicted_label.argmax(1))
            # print('ground truth: ', label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


def predict(file_path, model):
    print("Attempting prediction on given file using the following model: ")
    model.eval()
    if not exists(file_path):
        print(f'{file_path} does not exists')
        return None

    predict_dataset = JobPostingDataSet(file_path)
    dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    total_acc, total_count = 0, 0
    with torch.no_grad():
        #accuracy = torchmetrics.Accuracy().to(torch.device("cuda", 0))
        f1 = torchmetrics.F1Score().to(device)
        precision = torchmetrics.AveragePrecision().to(device)
        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            predictions = torch.argmax(predicted_label, 1)
            #acc_score = accuracy(labels, label)
            f1_score = f1(predictions, label)
            prec_score = precision(predictions, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    acc_result = total_acc / total_count
    print(f'prediction accuracy: {acc_result:.3f}')
    #acc_score = accuracy.compute()
    f1_score = f1.compute()
    print(f"The F1 score is {f1_score}")
    prec_score = precision.compute()
    print(f"The precision score is {prec_score}")
    return acc_result


def predict_instance(model, text, _text_pipeline):
    # output 0 or 1 for real or fake respectively
    with torch.no_grad():
        text = torch.tensor(_text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()


def rest_LR(model):
    global optimizer, scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)


def train_valid_test(model, _train_dataloader, _valid_dataloader, _test_dataloader):
    # perform one cycle of training (multiple batches), validation and test
    total_accu = 0
    rest_LR(model)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(_train_dataloader, model, epoch)
        accu_val = evaluate(_valid_dataloader, model)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print("\nPredictions: \n")
        print('-' * 59)

    print('Checking the results of test dataset.')
    accu_test = evaluate(_test_dataloader, model)
    print('test accuracy {:8.3f}'.format(accu_test))

    return accu_test


def simple_trainer(dataset, model, to_save=False):
    # simple training by splitting dataset into training, validation and testing
    num_test = int(len(dataset) * test_ratio)
    train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test])
    num_train = int(len(train_dataset) * train_ratio)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    train_valid_test(model, train_dataloader, valid_dataloader, test_dataloader)

    if to_save:
        save_model(model)


def k_folds_trainer(dataset, model, k, to_save=False):
    # k-folds training, model with the best accuracy is saved
    fold_size = int(len(dataset) / k)
    folds = random_split(dataset, [fold_size] * k)
    max_model = 0, None

    for idx, fold in enumerate(folds):
        print('-' * 59)
        print(f'Fold {idx + 1}')

        temp_model = copy.deepcopy(model)
        test_dataset = fold
        train_dataset = ConcatDataset([f for f in folds if f != fold])
        num_train = int(len(train_dataset) * train_ratio)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

        acc = train_valid_test(temp_model, train_dataloader, valid_dataloader, test_dataloader)
        max_model = max(max_model, (acc, temp_model), key=lambda x: x[0])

    if to_save:
        save_model(max_model[1])

def main():
    # TextClassificationModel variables
    num_class = 2  # num of labels, (e.g. fraudulent variable only takes on two value)
    vocab = ds.vocab_list.get('combined_description')
    vocab_size = len(vocab)
    emsize = 128

    # will load from previous model from model_weights.pth
    # delete model_weights.pth to train from new model
    tc_model = load_model()
    if not tc_model:
        print('new model created')
        tc_model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    else:
        print('model loaded')
    job_label = {0: 'Real', 1: 'Fake'}

    # Hyperparameters
    global EPOCHS, LR, BATCH_SIZE, test_ratio, train_ratio
    EPOCHS = 10  # epoch
    LR = 5  # learning rate
    BATCH_SIZE = 64  # batch size for training
    test_ratio = 0.1
    train_ratio = 0.95

    # Model Training Functions
    global criterion, optimizer, scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(tc_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    k_folds_trainer(ds, tc_model, k=10, to_save=True)
    predict('random_sample.csv', tc_model)

if __name__ == "__main__":
    main()
