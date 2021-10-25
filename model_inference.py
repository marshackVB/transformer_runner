# Databricks notebook source
# MAGIC %md ### Performing inference using a fitted model from the Registry

# COMMAND ----------

# MAGIC %md Pip install dependencies stored with the model, restart the Python environment, and import full suite of dependecies

# COMMAND ----------

dbutils.widgets.text("model_name",  '')
dbutils.widgets.text("stage",       '')

# Expects database.table_name
dbutils.widgets.text("input_df",    '')
dbutils.widgets.text("output_df",   '')

dbutils.widgets.text("feature_col", '')
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
                          "model/requirements.txt", 
                          "/usr/")

%pip install -q -r /usr/model/requirements.txt

# COMMAND ----------

from helpers import get_run_id
import mlflow

# COMMAND ----------

model_name =      dbutils.widgets.get("model_name")
stage =           dbutils.widgets.get("stage")
input_df =        dbutils.widgets.get("input_df")
feature_col =     dbutils.widgets.get("feature_col")
output_df =       dbutils.widgets.get("output_df")

# COMMAND ----------

# MAGIC %md ### Load production model from registry and apply to new data

# COMMAND ----------

run_id = get_run_id(model_name, stage=stage)

registered_model = f'runs:/{run_id}/model'

loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=registered_model)

print(f"This model was fit using the following runtime: {loaded_model.metadata.databricks_runtime}")

df = spark.table(input_df)

predictions = df.withColumn('predictions', loaded_model(feature_col))

# COMMAND ----------

if output_df:
  spark.write.mode('overwrite').format('delta').saveAsTable(output_df)

else:
  display(predictions)
