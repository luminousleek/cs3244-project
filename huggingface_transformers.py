from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
# import logging

dataset = load_dataset('csv', data_files={'train': 'train_data_reduced.csv', 'test': 'test_data.csv'})
acc_metric = load_metric('accuracy')
f1_metric = load_metric('f1')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def preprocess_function(examples):
    examples['text'] = [str(text) for text in examples['text']]
    tokenized_batch = tokenizer(examples['text'], padding=True, truncation=True, max_length=128)
    tokenized_batch["labels"] = [int(label) for label in examples["labels"]]
    return tokenized_batch


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = acc_metric.compute(predictions=predictions, references=labels)
    f1_score = f1_metric.compute(predictions=predictions, references=labels)
    return {"accuracy": accuracy, "f1 score": f1_score}


tokenized_df = dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_df["train"],
    eval_dataset=tokenized_df["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
result = trainer.evaluate()
print(result)
# model.save_pretrained(save_directory="./models")