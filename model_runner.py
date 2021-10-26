# Databricks notebook source
# MAGIC %md ## Transformer models training notebook

# COMMAND ----------

# MAGIC %md Install dependecies

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt

# COMMAND ----------

import math
import yaml
from functools import partial
from sys import version_info
import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from pyspark.sql.functions import col
from sklearn.metrics import precision_recall_fscore_support
import mlflow
from mlflow.types import ColSpec, DataType, Schema
from custom_classes import TransformerIterableDataset, TransformerModel
from helpers import get_config, get_parquet_files, get_or_create_experiment, get_best_metric

# COMMAND ----------

# MAGIC %md Import yaml configs

# COMMAND ----------

dbutils.widgets.text("config_file", 'config.yaml')
config_file = dbutils.widgets.get("config_file")

config = get_config(config_file)

config.max_token_length = None if config.max_token_length == -1 else config.max_token_length

# COMMAND ----------

# MAGIC %md Configure MLflow tracking servier location

# COMMAND ----------

get_or_create_experiment(config.experiment_location)

# COMMAND ----------

# MAGIC %md Get lists of parquet files

# COMMAND ----------

train_files = get_parquet_files(config.database_name, config.train_table_name)
test_files = get_parquet_files(config.database_name, config.test_table_name)

# COMMAND ----------

# MAGIC %md Download and configure tokenizer and pretrained transformer model

# COMMAND ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(config.model_type)
model = AutoModelForSequenceClassification.from_pretrained(config.model_type, num_labels=config.num_labels).to(device)

# COMMAND ----------

# MAGIC %md Configure batch or streaming datasets

# COMMAND ----------

if config.streaming_read:
  
  # Determine max steps
  training_records = spark.table(f'{config.database_name}.{config.train_table_name}').count()

  parrallelism = 1 if device.type == 'cpu' else torch.cuda.device_count()
  gradient_accumulation_steps = 1
  
  effective_batch_size = config.batch_size * gradient_accumulation_steps * parrallelism
  max_steps = math.floor(config.num_train_epochs * training_records / effective_batch_size)
  
  # Since streaming datasets require a "steps" based evaluation strategy, 
  # calculate the eval steps required such that evaluation happens once
  # per epoch.
  eval_steps_for_epoch = max_steps / config.num_train_epochs
  
  
  load_train = load_dataset("parquet", 
                             data_files=train_files, 
                             split='train',
                             streaming=True)

  load_test = load_dataset("parquet", 
                            data_files=test_files, 
                            split='train',
                            streaming=True)

  train = TransformerIterableDataset(load_train, 
                                     tokenizer, 
                                     config.feature_col, 
                                     config.label_col, 
                                     config.max_token_length)

  test = TransformerIterableDataset(load_test, 
                                    tokenizer, 
                                    config.feature_col, 
                                    config.label_col, 
                                    config.max_token_length)
  
  train_test = DatasetDict({'train': train,
                            'test': test})
  
else:
    
  train = load_dataset("parquet", 
                        data_files=train_files,
                        split='train')

  test = load_dataset("parquet", 
                       data_files=test_files,
                       split='train')
  
  train_test = DatasetDict({'train': train,
                            'test': test})
  
  def tokenize(batch, feature_col=config.feature_col):
    return tokenizer(batch[feature_col], 
                     padding='max_length', 
                     truncation=True, 
                     max_length=config.max_length)
  
  train_test = train_test.map(tokenize, batched=True, batch_size=config.batch_size) 
  
  train_test.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])

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

# Batch configuration
training_params =   {"output_dir": '/Users/marshall.carter/Documents/huggingface/results',
                     "overwrite_output_dir":       True,
                     "per_device_train_batch_size": config.batch_size,
                     "per_device_eval_batch_size":  config.batch_size,
                     "weight_decay":                0.01,
                     "num_train_epochs":            config.num_train_epochs,
                     "save_strategy":               "epoch", 
                     "evaluation_strategy":         "epoch",
                     
                     # When set to True, the parameters save_strategy needs to be the same as eval_strategy, 
                     # and in the case it is “steps”, save_steps must be a round multiple of eval_steps
                     
                     "load_best_model_at_end":      True,
                     "save_total_limit":            config.save_total_limit,
                     "metric_for_best_model":       config.metric_for_best_model,
                     "greater_is_better":           True,
                     "seed":                        config.seed,
                     "report_to":                   'none'
                    }


# Adjusted for streaming
if config.streaming_read:
  # max_steps overrides num_training_epochs
  training_params['max_steps'] =                     max_steps
  
  # Evaluation is done (and logged) every eval_steps
  training_params['evaluation_strategy'] =           "steps"
  training_params['save_strategy'] =                 "steps"
  training_params['load_best_model_at_end'] =        True
  training_params['eval_steps'] =                    eval_steps_for_epoch
  training_params['save_steps'] =                    eval_steps_for_epoch
  
  
  
training_args = TrainingArguments(**training_params)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_test['train'],
                  eval_dataset=train_test['test'],
                  compute_metrics=compute_metrics)

# COMMAND ----------

# MAGIC %md Train the model. Log tokenizer and fitted model to MLflow. Log a custom PythonModel for inference in an MLflow sub-run

# COMMAND ----------

with mlflow.start_run(run_name=config.model_type) as run:
  
  # Train model
  trainer.train()
  
  # Log metrics
  get_metric = partial(get_best_metric, trainer.state.log_history)
  
  metrics_to_log = ['eval_f1', 'eval_precision', 'eval_recall', 'train_runtime', 'eval_runtime',
                    'eval_loss', 'train_loss']
  
  for metric in metrics_to_log:
    mlflow.log_metric(*get_metric(metric))
    
  python_version = "{major}.{minor}.{micro}".format(major=version_info.major,
                                                    minor=version_info.minor,
                                                    micro=version_info.micro)
  # Log parameters
  params = {"eval_batch_size":        trainer.args.eval_batch_size,
            "train_batch_size":       trainer.args.train_batch_size,
            "gpus":                   trainer.args._n_gpu,
            "epochs":                 trainer.args.num_train_epochs,
            "metric_for_best_model":  trainer.args.metric_for_best_model,
            "best_checkpoint":        trainer.state.best_model_checkpoint.split('/')[-1],
            "streaming_read":         config.streaming_read,
            "runtime_version":        spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"),
            "python_version":         python_version}
    
  mlflow.log_params(params)
                  
  # Log artifacts
  trainer.save_model('/test_model')
  tokenizer.save_pretrained('/test_tokenizer')

  mlflow.log_artifacts('/test_tokenizer', artifact_path='tokenizer')
  mlflow.log_artifacts('/test_model', artifact_path='model')
  mlflow.log_artifact('config.yaml', artifact_path='config')

  # Create a sub-run / child run that logs the custom inference class to MLflow
  with mlflow.start_run(run_name = "python_model", nested=True) as child_run:

      # Create custom model
      transformer_model = TransformerModel(tokenizer = '/test_tokenizer', model = '/test_model')
      
      # Create conda environment
      with open('requirements.txt', 'r') as additional_requirements:
        libraries = additional_requirements.readlines()
        libraries = [library.rstrip() for library in libraries]

      model_env = mlflow.pyfunc.get_default_conda_env()
      model_env['dependencies'][-1]['pip'] += libraries
      
      # Create model input and output schemas
      input_schema = Schema([ColSpec(name=config.feature_col,  type= DataType.string)])

      output_schema = Schema([ColSpec(name=config.feature_col, type= DataType.string),
                              ColSpec(name='prediction',       type= DataType.double)])

      signature = mlflow.models.ModelSignature(input_schema, output_schema)
      
      # Log custom model, signature, and conda environment
      mlflow.pyfunc.log_model("model", python_model=transformer_model, signature=signature, conda_env=model_env)

  mlflow.end_run()
