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

      apply_softmax = lambda x: softmax(x).numpy()

      apply_rounding = lambda x: np.around(x, decimals=4)

      model_input = model_input.iloc[:, 0].apply(apply_tokenizer)
      model_input = model_input.apply(apply_model)
      model_input = model_input.apply(apply_softmax)
      model_input = model_input.apply(apply_rounding)
    
    return model_input
  
  
class TransformerModelGPU(mlflow.pyfunc.PythonModel):
  def __init__(self, tokenizer, model, max_token_length):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.tokenizer = tokenizer
    self.model = model.to(self.device)
    self.max_token_length = max_token_length
    

  def predict(self, context, model_input):
    
    apply_tokenizer = lambda x: self.tokenizer(str(x), padding='max_length', truncation=True, max_length=self.max_token_length, return_tensors='pt') 

    with torch.no_grad():
      apply_model = lambda x: self.model(x['input_ids'].to(self.device), x['attention_mask'].to(self.device)).logits

      softmax = torch.nn.Softmax(dim=1)

      apply_softmax = lambda x: softmax(x).detach().to('cpu').numpy()

      apply_rounding = lambda x: np.around(x, decimals=4)

      model_input = model_input.iloc[:, 0].apply(apply_tokenizer)
      model_input = model_input.apply(apply_model)
      model_input = model_input.apply(apply_softmax)
      model_input = model_input.apply(apply_rounding)
    
    return model_input
      
      
      