import copy
import time
from os.path import exists

from torch import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from dataset import JobPostingDataSet, build_vocab
import torchmetrics

from model import collate_batch, dataset as ds, device, TextClassificationModel, save_model, load_model, text_pipeline

import wandb
# at beginning of the script

wandb.init(project="cs3244-project", entity="isaacleexj", config={})


def train(dataloader, model, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        wandb.log({"loss": loss})
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


def predict_with_result(file_path, to_train=True):
    model, vocab = initialise_model()
    model.eval()
    if not exists(file_path):
        print(f'{file_path} does not exists')
        return None

    predict_dataset = JobPostingDataSet(file_path)

    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (text, label) in enumerate(predict_dataset):
            actual_label = int(label)
            t_tensor = torch.tensor(text_pipeline(text, vocab)).to(device)
            o_tensor = torch.tensor([0]).to(device)
            predicted_label = model(t_tensor, o_tensor).argmax(1).item()

            if actual_label == 1 or actual_label != predicted_label:
                print(f'{idx}) actual/predicted:{actual_label}/{predicted_label}: {text}')
            else:
                total_acc += 1
            total_count += 1

    acc_result = total_acc / total_count
    print(f'prediction accuracy: {acc_result:.3f}')

    wandb.log({"prediction accuracy:": acc_result})

    if to_train:
        model.train()
    return acc_result


def predict(file_path):
    model, vocab = initialise_model()
    if model is None:
        print('unable to load model')
        return

    model.eval()
    if not exists(file_path):
        print(f'{file_path} does not exists')
        return None

    print(f'Prediction for {file_path}')

    predict_dataset = JobPostingDataSet(file_path)
    dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))

    total_acc, total_count = 0, 0
    with torch.no_grad():
        # accuracy = torchmetrics.Accuracy().to(torch.device("cuda", 0))
        f1 = torchmetrics.F1Score().to(device)
        precision = torchmetrics.AveragePrecision().to(device)

        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            predictions = torch.argmax(predicted_label, 1)
            # acc_score = accuracy(labels, label)

            f1(predictions, label)
            precision(predictions, label)

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

    acc_result = total_acc / total_count
    print(f'prediction accuracy: {acc_result:.3f}')

    # acc_score = accuracy.compute()
    f1_score = f1.compute()
    wandb.log({"prediction accuracy:": acc_result})
    print(f"The F1 score is {f1_score}")
    prec_score = precision.compute()
    print(f"The precision score is {prec_score}\n")
    return acc_result


def predict_instance(model, text, _text_pipeline):
    # output 0 or 1 for real or fake respectively

    with torch.no_grad():
        text = torch.tensor(_text_pipeline(text, model[1]))
        output = model[0](text, torch.tensor([0]))
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
    tc_model, vocab = load_model()
    if not tc_model:
        if vocab_size is None:
            print('no model returned')
            return None, None

        print('new model created')
        tc_model = TextClassificationModel(vocab_size, em_size, num_class)
        vocab = None
    else:
        print('model loaded')

    return tc_model.to(device), vocab


def simple_trainer(dataset, to_save=False):
    # simple training by splitting dataset into training, validation and testing
    f_ds, t_ds = dataset
    ft_ds = ConcatDataset([f_ds, t_ds])
    num_test = int(len(ft_ds) * test_ratio)
    train_dataset, test_dataset = random_split(ft_ds, [len(ft_ds) - num_test, num_test])
    num_train = int(len(train_dataset) * train_ratio)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    vocab = build_vocab(train_dataset)
    model, _ = initialise_model(len(vocab))

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch(vocab))

    train_valid_test(model, train_dataloader, valid_dataloader, test_dataloader)

    if to_save:
        save_model((model, vocab))


def k_folds_trainer(dataset, k, to_save=False):
    # k-folds training, model with the best accuracy is saved
    if k == 1:
        return simple_trainer(dataset, to_save)

    f_ds, t_ds = dataset
    ft_ratio = int(len(t_ds) / len(f_ds))
    fold_size = int(len(f_ds) / k)
    f_rest, t_rest = len(f_ds) - fold_size * k, len(t_ds) - fold_size * k * ft_ratio

    f_folds = random_split(f_ds, [fold_size] * k + [f_rest])
    t_folds = random_split(t_ds, [fold_size * ft_ratio] * k + [t_rest])
    f_folds.pop()
    t_folds.pop()

    max_model = 0, None, None
    loaded_model, _ = initialise_model()

    for idx, fold in enumerate(tuple(zip(f_folds, t_folds))):
        print('-' * 59)
        print(f'Fold {idx + 1}')

        test_dataset = ConcatDataset([fold[0]] + [fold[1]])
        train_dataset = ConcatDataset([f for f in f_folds if f != fold[0]] + [f for f in t_folds if f != fold[1]])
        num_train = int(len(train_dataset) * train_ratio)
        split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

        vocab = build_vocab(split_train_)
        if loaded_model is None:
            temp_model, _ = initialise_model(len(vocab))
        else:
            temp_model = copy.deepcopy(loaded_model)

        train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_batch(vocab))
        valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                      shuffle=True, collate_fn=collate_batch(vocab))
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                     shuffle=True, collate_fn=collate_batch(vocab))

        acc = train_valid_test(temp_model, train_dataloader, valid_dataloader, test_dataloader)
        max_model = max(max_model, (acc, temp_model, vocab), key=lambda x: x[0])
        wandb.log({"acc": acc, "fold": idx + 1})

    if to_save:
        save_model(max_model[1:])


# TextClassificationModel variables
num_class = 2  # num of labels, (e.g. fraudulent variable only takes on two value)
em_size = 128


job_label = {0: 'Real', 1: 'Fake'}

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate

BATCH_SIZE = 64  # batch size for training
test_ratio = 0.4
train_ratio = 0.8

wandb.config.update({
  "data": "description",
  "learning_rate": LR,
  "epochs": EPOCHS,
  "batch_size": BATCH_SIZE
})

# Model Training Functions
criterion = torch.nn.CrossEntropyLoss()

k_folds_trainer(ds, k=5, to_save=True)

predict('scraped_predicted_1.csv')
predict('fake_job_postings.csv')
predict('random_sample.csv')
predict('fake_postings_only.csv')
