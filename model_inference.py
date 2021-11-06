# Databricks notebook source
# MAGIC %md ### Performing inference using a fitted model from the Registry

# COMMAND ----------

# MAGIC %md Pip install dependencies stored with the model, restart the Python environment, and import full suite of dependecies

# COMMAND ----------

dbutils.widgets.text("model_name",  '')
dbutils.widgets.text("stage",       '')

dbutils.widgets.text("input_database",  'default')
dbutils.widgets.text("input_table",     'banking77_train')

dbutils.widgets.text("output_database", 'default')
dbutils.widgets.text("output_table",     '')

dbutils.widgets.text("feature_col", '')
dbutils.widgets.text("batch_size", '1000')

model_name =      dbutils.widgets.get("model_name")
stage =           dbutils.widgets.get("stage")

# COMMAND ----------

from mlflow.tracking import MlflowClient
from helpers import get_run_id
import mlflow

# Helper function to query registry
run_id = get_run_id(model_name, stage=stage)

client = MlflowClient()

# Copy requirements.txt from MLflow to driver
client.download_artifacts(run_id, 
                          "mlflow_model/requirements.txt", 
                          "/usr")

%pip install -q -r /usr/mlflow_model/requirements.txt

# COMMAND ----------

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql.types import ArrayType, FloatType
import pyspark.sql.functions as func
from pyspark.sql.functions import col
from helpers import get_run_id, get_parquet_files, get_config
import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

model_name =         dbutils.widgets.get("model_name")
stage =              dbutils.widgets.get("stage")

input_database =     dbutils.widgets.get("input_database")
input_table =        dbutils.widgets.get("input_table")

output_database =    dbutils.widgets.get("output_database")
output_table =       dbutils.widgets.get("output_table")

feature_col =        dbutils.widgets.get("feature_col")
batch_size =         int(dbutils.widgets.get("batch_size"))

# COMMAND ----------

# MAGIC %md Load production model from registry and apply to new data

# COMMAND ----------

run_id = get_run_id(model_name, stage)

client = MlflowClient()

# Copy artifacts to driver
client.download_artifacts(run_id=run_id, path="config", dst_path='/')
client.download_artifacts(run_id=run_id, path="huggingface_tokenizer", dst_path='/')
client.download_artifacts(run_id=run_id, path="huggingface_model", dst_path='/')

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = get_config('/config/config.yaml')
tokenizer = AutoTokenizer.from_pretrained('/huggingface_tokenizer')
model = AutoModelForSequenceClassification.from_pretrained('/huggingface_model').to(device)

# COMMAND ----------

# MAGIC %md Create data loader

# COMMAND ----------

input_files = get_parquet_files(input_database, input_table)
load_train = load_dataset("parquet", data_files=input_files)

# COMMAND ----------

# MAGIC %md Forward pass for inference

# COMMAND ----------

def forward_pass(batch, tokenizer=tokenizer, model=model, max_length=config.max_token_length, feature_col=feature_col):
  
  tokenized = tokenizer(batch[feature_col], 
                        padding='max_length', 
                        truncation=True, 
                        max_length=max_length)
  
  input_ids = torch.tensor(tokenized['input_ids']).to(device)
  attention_mask = torch.tensor(tokenized['attention_mask']).to(device)
  
  model = nn.DataParallel(model.cuda())
  
  with torch.no_grad():
    predictions = model(input_ids, attention_mask).logits

  softmax = torch.nn.Softmax(dim=1)
  predictions = softmax(predictions).detach().cpu().numpy()
   
  batch['probabilities'] = predictions
  
  return batch

# COMMAND ----------

# MAGIC %md Batch in records, apply model, and write final predictions to a Delta table

# COMMAND ----------

predictions = load_train.map(forward_pass, batched=True, batch_size=batch_size)

predictions.set_format(type='pandas')
predictions = predictions['train'][:]

predictions_df = (spark.createDataFrame(predictions)
                       .selectExpr(['*', 'array_max(probabilities) as predicted_probability'])
                       .selectExpr(['*', 'array_position(probabilities, predicted_probability) - 1 as predicted_class']))

# Can change to 'append' to add to a table of predictions
predictions_df.write.mode('overwrite').format('delta').saveAsTable(f'{output_database}.{output_table}')

display(predictions_df)
