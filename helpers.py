import yaml
import json
from argparse import Namespace
from typing import List, Tuple, Dict
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
import mlflow
from mlflow.tracking import MlflowClient
from transformers import Trainer

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

client = MlflowClient()


def get_config(config_file:str) -> Namespace:
  """
  Loads paramerts from a yaml configuration file in a
  Namespce object.

  Args:
    config_file: A path to a yaml configuration file.

  Returns:
    A Namespace object that contans configurations referenced
    in the program.
  """
  stream = open(config_file, 'r')
  config_dict = yaml.load(stream, yaml.SafeLoader)
  
  for parameter, value in config_dict.items():
    print("{0:30} {1}".format(parameter, value))
    
  config = Namespace(**config_dict)
  
  return config


def get_parquet_files(database:str, table_name:str) -> List[str]:
  """
  Given a database and name of a parquet table, return a list of 
  parquet files paths and names that can be read by the transfomers
  library

  Args:
    database: The database where the table resides.
    table_name: The name of the table.

  Returns:
    A Namespace object that contans configurations referenced
    in the program.
  """
  
  files = spark.table(f'{database}.{table_name}').inputFiles()
  
  if files[0][:4] == 'dbfs':
    files = [file.replace('dbfs:', '/dbfs/') for file in files]
  
  return files


def get_best_metrics(trainer: Trainer) -> Dict[str, float]:
  """
  Extract metrics from a fitted Trainer instance.

  Args:
    trainer: A Trainer instance that has been trained on data.
   
  Returns:
    A dictionary of metrics and their values.
  """

  # Best model metrics
  best_checkpoint = f'{trainer.state.best_model_checkpoint}/trainer_state.json' 

  with open(best_checkpoint) as f:
    metrics = json.load(f)

  best_step = metrics['global_step']

  all_log_history = enumerate(metrics['log_history'])

  best_log_idx = [idx for idx, values in all_log_history if values['step'] == best_step][0]

  best_log = metrics['log_history'][best_log_idx]
  best_log.pop('epoch')

  # Overal runtime metrics
  runtime_logs_idx = [idx for idx, values in enumerate(trainer.state.log_history) if values.get('train_runtime') is not None][0]
  runtime_logs = trainer.state.log_history[runtime_logs_idx]

  best_log['train_runtime'] = runtime_logs['train_runtime']
  best_log['train_loss'] = runtime_logs['train_loss']



  return best_log

      
def get_or_create_experiment(experiment_location: str) -> None:
  """
  Given an experiement path, check to see if an experiment exists in the location.
  If not, create a new experiment. Set the notebook to log all experiments to the
  specified experiment location

  Args:
    experiment_location: The path to the MLflow Tracking Server (Experiement) instance, 
                         viewable in the upper left hand corner of the server's UI.
  """

  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)
  
  
def get_run_id(model_name:str, stage:str='Production') -> str:
  """Given a model's name, return its run id from the Model Registry; this assumes the model
  has been registered
  
  Args:
    model_name: The name of the model; this is the name used to registr the model.
    stage: The stage (version) of the model in the registr you want returned
    
  Returns:
    The run id of the model; this can be used to load the model for inference
  
  """
  
  prod_run = [run for run in client.search_model_versions(f"name='{model_name}'") 
                  if run.current_stage == stage][0]
  
  return prod_run.run_id