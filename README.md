# Rapid NLP development with Databricks, Delta, and Transformers

This Databricks [Repo](https://docs.databricks.com/repos.html) provides a ready-to-use workflow for applying a broad array of [pre-trained transformers](https://huggingface.co/transformers/pretrained_models.html) for text classification tasks on Delta tables in your Delta Lake.  

Only a Databricks Workspace, and two Delta tables - one for training, the other for testing, are required. The Delta tables should consist of one text column (the feature) and one label column (the values to predict).

If you already have a Delta table ready to go, it's easy to generate the above dependencies using the below code (replacing the database and table names to your versions). The column you're predicting should be an integer type and start at value 0.

```python
df = (spark.table('default.text_classification_table')
           .selectExpr('text', 
                       'cast(label as integer) as label'))

train, test = df.randomSplit([0.7, 0.3], seed=12345)

train.write.mode('overwrite').format('delta').saveAsTable('default.train')
test.write.mode('overwrite').format('delta').saveAsTable('default.test')
```

The Repo can read the tables' underlying parquet files directly, either as a stream using pytorchs' [IterableDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset) or using the traditional [map-style dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets).  

## Running the Repo on your data  
#### Steps:    
1. Ensure that [non-notebook file support](https://docs.databricks.com/repos.html#work-with-non-notebook-files-in-a-databricks-repo) is enable in your Workspace.  

2. Create a new Repo in your Databricks Workspace and clone this repository into the Repo.  

3. Provision a single-node cluster, preferably using a GPU instance type for much better performance. This project was most recently tested with [runtime version 9.0 ML](https://docs.databricks.com/release-notes/runtime/9.0.html)  

4. Edit the config.yaml file to your specification. 
    - Change the database, table, and column name variables to your versions.  

    - Ensure 'num_labels' matches the number of distinct labels in your data.  

    - Enter the path to an MLflow Tracking Server. If a server is not found at the path, a new server instance will be created in that location.  

    - If 'streaming_read' is set to True, your parquet files will be read as a stream; if set to false, they will be read as a traditional pytorch dataset.  

    - The other parameters govern various aspects of model training and are typically documented in the [Trainer](https://huggingface.co/transformers/main_classes/trainer.html#trainer) and [TrainingArguments](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments) secions of the [transformers documentation](https://huggingface.co/transformers/index.html). More parameters can be added to the config.yaml file for more flexibility.  

5. Open the **model_runner** notebook and select **Run All** at the top of the UI.  

Each time you run the model_runner notebook, a new entry will be logged to the tracking server, giving you the ability to try different model types and parameters and compare performance across runs. You can also schedule the model_runner notebook to run as a job for concurrent runs - just create an additional yaml file for each run. When you create the job, add a Parameter called 'config_file' pointing to the correct yaml file for your run.  

## Performing inference  

After you have found a performant model, you can log that model version in the Model Registry. The inference notebook provides an easy workflow for loading a model from the registry and apply it's predict method to another Delta table.

The inference notebook accepts the following parameters.
 - **model_name**: The name of the registered model.
 - **stage**: The model's stage in the registry, such as 'Production'.
 - **input_df**: The Delta table on which to apply the model (expects 'database.table_name').
 - **feature_col**: The name of the text column that will be input to the model.
 - **output_df**: The Delta table where results will be writen (expects 'database.table_name').







