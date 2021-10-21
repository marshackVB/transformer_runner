# Databricks notebook source
# MAGIC %md ### Transformer models training notebooks 

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import math
import yaml
import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pyspark.sql.functions import col
from sklearn.metrics import precision_recall_fscore_support
import mlflow
from custom_classes import TransformerIterableDataset, TransformerModel

# COMMAND ----------

dbutils.widgets.text("config_file_name", '')
config_file_name = dbutils.widgets.get("config_file_name")

# COMMAND ----------

stream = open(config_file_name, 'r')
config_dict = yaml.load(stream, yaml.SafeLoader)

for parameter, value in config_dict.items():
  print("{0:30} {1}".format(parameter, value))

class dotdict(dict):
    __getattr__ = dict.get

config = dotdict(config_dict)

# COMMAND ----------

# MAGIC %md Configure MLflow tracking servier location

# COMMAND ----------

mlflow.set_experiment(config.experiment_location)

# COMMAND ----------

# MAGIC %md Get lists of parquet files

# COMMAND ----------

parquet_base_dir_py = f"/{config.parquet_base_dir.replace(':', '')}"

train_files = [file.path for file in dbutils.fs.ls(f'{config.parquet_base_dir}{config.train_table_name}') if file.path[-8:] == '.parquet']
train_files = [file.replace(config.parquet_base_dir, parquet_base_dir_py) for file in train_files]
 
test_files = [file.path for file in dbutils.fs.ls(f'{config.parquet_base_dir}{config.test_table_name}') if file.path[-8:] == '.parquet']
test_files = [file.replace(config.parquet_base_dir, parquet_base_dir_py) for file in test_files]

# COMMAND ----------

# MAGIC %md Configure max_steps based requested epochs and other information

# COMMAND ----------

training_records = spark.read.parquet(f'{config.parquet_base_dir}{config.train_table_name}').count()

num_gpus = torch.cuda.device_count()
gradient_accumulation_steps = 1
effective_batch_size = config.batch_size * gradient_accumulation_steps * num_gpus
max_steps = math.floor(config.num_training_epochs * training_records / effective_batch_size)

# COMMAND ----------

# MAGIC %md Download and configure tokenizer and pretrained transformer model

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained(config.model_type)
model = AutoModelForSequenceClassification.from_pretrained(config.model_type, num_labels=config.num_labels)

# COMMAND ----------

# MAGIC %md Create streaming datasets

# COMMAND ----------

stream_train = load_dataset("parquet", 
                            data_files=train_files, 
                            split='train', 
                            streaming=True)

stream_test = load_dataset("parquet", 
                           data_files=test_files, 
                           split='train', 
                           streaming=True)

train = TransformerIterableDataset(stream_train, 
                                   tokenizer, 
                                   config.feature_col, 
                                   config.label_col, 
                                   config.max_token_length)

test = TransformerIterableDataset(stream_test, 
                                  tokenizer, 
                                  config.feature_col, 
                                  config.label_col, 
                                  config.max_token_length)

# COMMAND ----------

# MAGIC %md Configure validation metrics and model trainer

# COMMAND ----------

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
            }


training_args = TrainingArguments(output_dir =                '/huggingface/results',
                                  overwrite_output_dir =       True,
                                  per_device_train_batch_size= config.batch_size,
                                  per_device_eval_batch_size=  config.batch_size,
                                  weight_decay=                0.01,
                                  max_steps =                  max_steps,
                                  save_strategy =              "steps", # The default
                                  evaluation_strategy =        "steps",
                                  #save_steps = 10, # Default is 500
                                  eval_steps =                 config.eval_steps,
                                  save_total_limit =           config.save_total_limit,
                                  load_best_model_at_end=      True,
                                  metric_for_best_model =      config.metric_for_best_model,
                                  greater_is_better =          True,
                                  seed=                        config.seed)

def get_trainer(model=model, args=training_args, train_dataset=train, eval_dataset=test, compute_metrics=compute_metrics):
  
  trainer = Trainer(model=model,
                    args=training_args,
                    train_dataset=train,
                    eval_dataset=test,
                    compute_metrics=compute_metrics)
  return trainer

trainer = get_trainer()

# COMMAND ----------

# MAGIC %md Train the model and log tokenizer and fitted model to MLflow

# COMMAND ----------

trainer.train()

trainer.save_model('/test_model')
tokenizer.save_pretrained('/test_tokenizer')
  
mlflow.log_artifacts('/test_tokenizer', artifact_path='tokenizer')
mlflow.log_artifacts('/test_model', artifact_path='model')
  
results = trainer.evaluate()

trainer.save_model('/test_model')
tokenizer.save_pretrained('/test_tokenizer')
  
mlflow.log_artifacts('/test_tokenizer', artifact_path='tokenizer')
mlflow.log_artifacts('/test_model', artifact_path='model')
mlflow.log_artifact('config.yaml', artifact_path='config')
  
results = trainer.evaluate()

# Create a sub-run / child run that logs the custom inference class to MLflow
with mlflow.start_run(run_name = "model", nested=True) as child_run:
  
    transformer_model = TransformerModel(tokenizer = '/test_tokenizer', model = '/test_model')
    mlflow.pyfunc.log_model("model", python_model=transformer_model)

# COMMAND ----------

mlflow.end_run()
