from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

from dataset import combine_descriptions

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# prep training data
job_posting = pd.read_csv("cleaned_job_postings.csv")
combine_descriptions(job_posting)
relevant_cols = [job_posting['combined_description'], job_posting['fraudulent']]
relevant_df = pd.concat(relevant_cols, axis=1, keys=['combined_description', 'fraudulent'])
train_df = relevant_df.sample(frac=0.9)
test_df = relevant_df.drop(train_df.index)

# model arguments
model_args = ClassificationArgs
model_args.num_train_epochs = 5
model_args.learning_rate = 0.01
model_args.train_batch_size = 64
model_args.eval_batch_size = 64
model_args.wandb_project = "cs3244-project"

# early stopping
model_args.use_early_stopping = True
model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "mcc"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 5
model_args.evaluate_during_training_steps = 1000

# create model
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test_df)
