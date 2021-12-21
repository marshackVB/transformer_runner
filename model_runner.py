# Databricks notebook source
# MAGIC %md ## Transformer models training notebook

# COMMAND ----------

# MAGIC %md Install dependecies

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt

# COMMAND ----------

import os
import math
import yaml
from sys import version_info
import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import mlflow
from mlflow.types import ColSpec, DataType, Schema
from pyspark.sql.types import StructType, StructField, ArrayType, FloatType, DoubleType, StringType
from custom_pyfuncs import TransformerModel
from helpers import get_config, get_parquet_files, get_or_create_experiment, get_best_metrics

# COMMAND ----------

# MAGIC %md Import yaml configs

# COMMAND ----------

dbutils.widgets.text("config_file", 'config.yaml')
config_file = dbutils.widgets.get("config_file")

config = get_config(config_file)

config.max_token_length = None if config.max_token_length == -1 else config.max_token_length

# COMMAND ----------

# MAGIC %md Configure MLflow tracking server location

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

config.streaming_read = True
if config.streaming_read:
# https://huggingface.co/docs/datasets/dataset_streaming.html
  
  # Determine max steps
  training_records = spark.table(f'{config.database_name}.{config.train_table_name}').count()

  parrallelism = 1 if device.type == 'cpu' else torch.cuda.device_count()
  gradient_accumulation_steps = 1
  
  effective_batch_size = config.batch_size * gradient_accumulation_steps * parrallelism
  max_steps = math.floor(config.num_train_epochs * training_records / effective_batch_size)
  
  # Since streaming datasets require a "steps" based evaluation strategy, 
  # calculate the eval steps required such that evaluation happens once
  # per epoch.
  eval_steps_for_epoch = math.floor(max_steps / config.num_train_epochs)
  
  
  # 'train' is the only option when splitting is not done via the
  # transformers library iteself. This is only a dictionary key so
  # no worries.
  load_train = load_dataset("parquet", 
                             data_files=train_files, 
                             split='train',
                             streaming=True)

  load_test = load_dataset("parquet", 
                            data_files=test_files, 
                            split='train',
                            streaming=True)
  
  def tokenize(text, label):
    """
    Tokenizer for streaming read applied to a datasets.IterableDataset.
    """
    tokenized = tokenizer(text, 
                          padding='max_length', 
                          truncation=True, 
                          max_length=config.max_token_length)

    tokenized['label'] = label

    return tokenized
    
    
  f = lambda x: tokenize(x[config.feature_col], x[config.label_col])
    
  # See docs regarding mapping a function to a dataset.IterableDataset at
  # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.IterableDataset
  train_tokenized = load_train.map(f)
  test_tokenized = load_test.map(f)
  
  # Note that this command is listed a experimental in the documentation... 
  # https://huggingface.co/docs/datasets/dataset_streaming.html#working-with-numpy-pandas-pytorch-and-tensorflow
  # The command converts the datasets.IterableDataset object to a torch.utils.data.IterableDataset comprised of torch
  # tensors. Without this command, it would be necessary to create a class that enherits from torch.utils.data.IterableDataset,
  # applies the tokenizer, and returns an iterator. An example is available at...
  # https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset. If this objct convertion is not done, the
  # training step will through an error
  train = train_tokenized.with_format("torch")
  test = test_tokenized.with_format("torch")
  
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
  
  def tokenize(batch):
    """
    Tokenizer for non-streaming read. Additional features are available when using
    the map function of a dataset.Dataset instead of a dataset.IterableDataset, 
    therefore different tokenizer functions are used for each case.
    """
    return tokenizer(batch[config.feature_col], 
                     padding='max_length', 
                     truncation=True, 
                     max_length=config.max_token_length)
  
  # See the docs for mapping a function to a DatasetDict at
  # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.DatasetDict.map
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
  
  best_metrics = get_best_metrics(trainer)
  
  mlflow.log_metrics(best_metrics)
    
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
                  
  # Log artifacts for batch processing
  model_dir = '/huggingface_model'
  trainer.save_model(model_dir)
  tokenizer.save_pretrained(model_dir)

  mlflow.log_artifacts(model_dir, artifact_path='huggingface_model')
  mlflow.log_artifact('config.yaml')

  # Create a sub-run / child run that logs the custom inference class to MLflow
  #with mlflow.start_run(run_name = "python_model", nested=True) as child_run:

  # Create custom model for REST API inference
  # The model must be copied to the CPU if the custom pytfunc model will served via REST API, otherwise
  # an error will be thrown when attempting to served because required CUDA dependencies are not installed
  # on the serving cluster.
  cpu_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to('cpu')
  transformer_model = TransformerModel(tokenizer = tokenizer,
                                       model = cpu_model,
                                       max_token_length= config.max_token_length)

  # Create conda environment
  with open('requirements.txt', 'r') as additional_requirements:
    libraries = additional_requirements.readlines()
    libraries = [library.rstrip() for library in libraries]

  model_env = mlflow.pyfunc.get_default_conda_env()
  model_env['dependencies'][-1]['pip'] += libraries

  input_example = (spark.table(f'{config.database_name}.{config.train_table_name}')
                      .select(config.feature_col)
                      .limit(5)).toPandas()
  
  mlflow.pyfunc.log_model("mlflow_model", 
                          python_model=transformer_model, 
                          conda_env=model_env,
                          code_path=[os.path.abspath('custom_pyfuncs.py')],
                          input_example=input_example)
