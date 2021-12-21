import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow

     
class TransformerModel(mlflow.pyfunc.PythonModel):
  def __init__(self, tokenizer, model, max_token_length):
    self.tokenizer = tokenizer
    self.model = model
    self.max_token_length = max_token_length
    

  def predict(self, context, model_input):
    
    apply_tokenizer = lambda x: self.tokenizer(str(x), padding='max_length', truncation=True, max_length=self.max_token_length, return_tensors='pt') 

    with torch.no_grad():
      apply_model = lambda x: self.model(x['input_ids'], x['attention_mask']).logits

      softmax = torch.nn.Softmax(dim=1)

      apply_softmax = lambda x: softmax(x).numpy().flatten()

      apply_rounding = lambda x: np.around(x, decimals=4)

      model_input = model_input.iloc[:, 0].apply(apply_tokenizer)
      model_input = model_input.apply(apply_model)
      model_input = model_input.apply(apply_softmax)
      model_input = model_input.apply(apply_rounding)
    
    return np.vstack(model_input.values)