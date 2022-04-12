from os.path import exists

import pandas
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

from dataset import combine_descriptions, split_data

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# prep training data
# fjp, tjp = tuple(map(combine_descriptions, split_data("cleaned_job_postings.csv")))
# fjp_size = len(fjp)
# ft_ratio = 5
# tjp = tjp.sample(n=fjp_size * ft_ratio)
# job_posting = pandas.concat([fjp, tjp], axis=0)
job_posting = pandas.read_csv("cleaned_job_postings.csv")
combine_descriptions(job_posting)

relevant_cols = [job_posting['combined_description'], job_posting['fraudulent']]
relevant_df = pd.concat(relevant_cols, axis=1, keys=['combined_description', 'fraudulent'])
train_df = relevant_df.sample(frac=0.8)
train_df.columns = ["text", "labels"]
test_df = relevant_df.drop(train_df.index)
test_df.columns = ["text", "labels"]


def predict_file(_model, file_path):
    df = pandas.read_csv(file_path)
    combine_descriptions(df)
    _result = _model.predict(df['combined_description'].tolist())
    acc, count = 0, len(_result[0])
    for predict, actual in tuple(zip(_result[0], df['fraudulent'].tolist())):
        print(predict, actual)
        acc += 1 if int(predict) == int(actual) else 0
    print(acc / count)


def driver(model, to_train=False, to_evaluate=False, to_predict=False, prediction_file=None):
    if to_train:
        # Train the model
        model.train_model(train_df, output_dir=save_file)

    if to_evaluate:
        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(test_df, output_dir=save_file)
    if to_predict and prediction_file is not None:
        # Predict with model
        predict_file(model, prediction_file)


if __name__ == "__main__":
    # model arguments
    model_args = ClassificationArgs()
    model_args.num_train_epochs = 5
    model_args.learning_rate = 0.001
    model_args.train_batch_size = 16
    model_args.eval_batch_size = 16
    model_args.wandb_project = "cs3244-project"
    model_args.use_multiprocessing = False

    # early stopping
    model_args.use_early_stopping = True
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "mcc"
    model_args.early_stopping_metric_minimize = False
    model_args.early_stopping_patience = 5
    model_args.evaluate_during_training_steps = 1000

    load_file = "outputs/temp10"
    save_file = "outputs/temp11"
    if not exists(load_file):
        load_file = "roberta-base"

    # create model
    model = ClassificationModel(
        "roberta", load_file, args=model_args, use_cuda=True
    )

    driver(model, to_train=True, to_evaluate=False, to_predict=True, prediction_file='random_sample.csv')
