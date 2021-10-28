import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow


class TransformerModel(mlflow.pyfunc.PythonModel):
  def __init__(self, tokenizer, model, max_token_length):
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    self.model = AutoModelForSequenceClassification.from_pretrained(model)
    self.max_token_length = max_token_length


  def predict(self, context, model_input):
    """From docs: model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list
    """

    # Convert model input df to list of inputs
    feature_col = model_input.columns[0]
    input_to_lst = model_input[feature_col].to_list()
    tokenized = self.tokenizer(input_to_lst, padding='max_length', truncation=True, max_length=self.max_token_length)

    logits = self.model(torch.tensor(tokenized['input_ids']), 
                        torch.tensor(tokenized['attention_mask'])).logits

    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)
    probs = probs.detach().numpy()
    probs = np.around(probs, decimals=4)

    return probs
      
      