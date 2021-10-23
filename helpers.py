import yaml
from argparse import Namespace
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
import mlflow

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)


def get_config(config_file):
  stream = open(config_file, 'r')
  config_dict = yaml.load(stream, yaml.SafeLoader)
  
  for parameter, value in config_dict.items():
    print("{0:30} {1}".format(parameter, value))
    
  config = Namespace(**config_dict)
  
  return config


def get_parquet_files(database, table_name):
  
  files = spark.table(f'{database}.{table_name}').inputFiles()
  
  if files[0][:4] == 'dbfs':
    files = [file.replace('dbfs:', '/dbfs/') for file in files]
  
  return files


def get_best_metric(model_state, metric, n_decimals=4):
  
  for train_eval in model_state:
    for metric_name, value in train_eval.items():
      if metric_name == metric:
        return [metric_name, round(value, n_decimals)]

      
def get_or_create_experiment(experiment_location):
  if not mlflow.get_experiment_by_name(experiment_location):
    print("Experiment does not exist. Creating experiment")
    
    mlflow.create_experiment(experiment_location)
    
  mlflow.set_experiment(experiment_location)