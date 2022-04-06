import time
from os.path import exists

from torch import torch
from torch.utils.data import DataLoader, random_split

from dataset import JobPostingDataSet
from model import collate_batch, dataset, device, TextClassificationModel, save_model, load_model


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
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count


def predict(file_path, model):
    model.eval()
    if not exists(file_path):
        print(f'{file_path} does not exists')
        return None

    predict_dataset = JobPostingDataSet(file_path)
    dataloader = DataLoader(predict_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    total_acc, total_count = 0, 0
    with torch.no_grad():
        for label, text, offsets in dataloader:
            predicted_label = model(text, offsets)
            criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)


def predict_instance(model, text, _text_pipeline):
    # output 0 or 1 for real or fake respectively
    with torch.no_grad():
        text = torch.tensor(_text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()


def rest_LR():
    global optimizer, scheduler
    optimizer = torch.optim.SGD(tc_model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)


def train_valid_test(_train_dataloader, _valid_dataloader, _test_dataloader):
    # perform one cycle of training (multiple batches), validation and test
    total_accu = None
    rest_LR()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(_train_dataloader, tc_model, epoch)
        accu_val = evaluate(_valid_dataloader, tc_model)
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
        accu_test = evaluate(_test_dataloader, tc_model)
        print('test accuracy {:8.3f}'.format(accu_test))


def simple_trainer(_dataset, _model, to_save=False):
    num_test = int(len(_dataset) * test_ratio)
    train_dataset, test_dataset = random_split(_dataset, [len(_dataset) - num_test, num_test])
    num_train = int(len(train_dataset) * train_ratio)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    train_valid_test(train_dataloader, valid_dataloader, test_dataloader)

    if to_save:
        save_model(_model)


# TextClassificationModel variables
num_class = 2  # num of labels, (e.g. fraudulent variable only takes on two value)
vocab = dataset.vocab_list.get('combined_description')
vocab_size = len(vocab)
emsize = 128

tc_model = load_model()
if not tc_model:
    print('new model created')
    tc_model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
else:
    print('model loaded')

job_label = {0: 'Real', 1: 'Fake'}

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
test_ratio = 0.1
train_ratio = 0.95

# Model Training Functions
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(tc_model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

simple_trainer(dataset, tc_model, to_save=True)
