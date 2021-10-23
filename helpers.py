import yaml
from argparse import Namespace
from typing import List, Tuple, Dict
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
import mlflow

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


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


def get_parquet_files(database:str, table_name:str) -> List(str):
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


def get_best_metric(model_state:List(Dict(str, float)), metric:str, n_decimals:int = 4) -> Tuple(str, float):
  """
  Given a metric name, return its value from the best model.

  Args: 
    model_state: The log history of the best model's state (trainer.state.log_history).
    metric: The metric of interest.
    n_decimals: The number of decimals for rounding

  Returns:
    The metric of interest and its value
  """

  for train_eval in model_state:
    for metric_name, value in train_eval.items():
      if metric_name == metric:
        return (metric_name, round(value, n_decimals))

      
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