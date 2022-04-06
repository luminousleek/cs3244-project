import copy
import time
from os.path import exists

from torch import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from dataset import JobPostingDataSet, build_vocab
from model import collate_batch, dataset as ds, device, TextClassificationModel, save_model, load_model


def train(dataloader, model, optimizer, epoch):
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
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


def predict(file_path):
    model = initialise_model()
    if model is None:
        print('unable to load model')
        return

    model.eval()
    if not exists(file_path):
        print(f'{file_path} does not exists')
        return None

    predict_dataset = JobPostingDataSet(file_path)
    vocab = build_vocab(predict_dataset)
    dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))

    total_acc, total_count = 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    acc_result = total_acc / total_count
    print(f'prediction accuracy: {acc_result:.3f}')
    return acc_result


def predict_instance(model, text, _text_pipeline):
    # output 0 or 1 for real or fake respectively
    with torch.no_grad():
        text = torch.tensor(_text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()


def get_optimizer_scheduler(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)
    return optimizer, scheduler


def train_valid_test(model, _train_dataloader, _valid_dataloader, _test_dataloader):
    # perform one cycle of training (multiple batches), validation and test
    total_accu = 0
    optimizer, scheduler = get_optimizer_scheduler(model)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(_train_dataloader, model, optimizer, epoch)
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
        print('-' * 59)

    print('Checking the results of test dataset.')
    accu_test = evaluate(_test_dataloader, model)
    print('test accuracy {:8.3f}'.format(accu_test))

    return accu_test


def initialise_model(vocab_size=None):
    # will load from previous model from model_weights.pth
    # delete model_weights.pth to train from new model
    tc_model = load_model()
    if not tc_model:
        if vocab_size is None:
            print('no model returned')
            return None

        print('new model created')
        tc_model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    else:
        print('model loaded')

    return tc_model


def simple_trainer(dataset, to_save=False):
    # simple training by splitting dataset into training, validation and testing
    num_test = int(len(dataset) * test_ratio)
    train_dataset, test_dataset = random_split(dataset, [len(dataset) - num_test, num_test])
    num_train = int(len(train_dataset) * train_ratio)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    vocab = build_vocab(train_dataset)
    model = initialise_model(len(vocab))

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))

    train_valid_test(model, train_dataloader, valid_dataloader, test_dataloader)

    if to_save:
        save_model(model)


def k_folds_trainer(dataset, k, to_save=False):
    # k-folds training, model with the best accuracy is saved
    fold_size = int(len(dataset) / k)
    folds = random_split(dataset, [fold_size] * k)
    max_model = 0, None
    loaded_model = initialise_model()

    for idx, fold in enumerate(folds):
        print('-' * 59)
        print(f'Fold {idx + 1}')

        test_dataset = fold
        train_dataset = ConcatDataset([f for f in folds if f != fold])
        num_train = int(len(train_dataset) * train_ratio)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        vocab = build_vocab(train_dataset)
        if loaded_model is None:
            temp_model = initialise_model(len(vocab))
        else:
            temp_model = copy.deepcopy(loaded_model)

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_batch(vocab))
        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_batch(vocab))
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                     shuffle=True, collate_fn=collate_batch(vocab))

        acc = train_valid_test(temp_model, train_dataloader, valid_dataloader, test_dataloader)
        max_model = max(max_model, (acc, temp_model), key=lambda x: x[0])

    if to_save:
        save_model(max_model[1])


# TextClassificationModel variables
num_class = 2  # num of labels, (e.g. fraudulent variable only takes on two value)
emsize = 128

job_label = {0: 'Real', 1: 'Fake'}

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
test_ratio = 0.1
train_ratio = 0.95

criterion = torch.nn.CrossEntropyLoss()

k_folds_trainer(ds, k=20, to_save=True)
predict('random_sample.csv')
