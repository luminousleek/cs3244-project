import torch
from dataset import JobPostingDataSet
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
filename = 'cleaned_job_postings.csv'

training_data = JobPostingDataSet(filename)
# test_data = JobPostingDataSet(filename)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

train_features, train_label = next(iter(train_dataloader))

if __name__ == "__main__":
    pass
