import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import mlflow


class TransformerIterableDataset(IterableDataset):
  def __init__(self, data, tokenizer, feature_col, label_col, max_length):
    super(TransformerIterableDataset).__init__()
    self.data = data
    self.tokenizer = tokenizer
    self.feature_col = feature_col
    self.label_col = label_col
    self.max_length = max_length

  def tokenize_stream(self):
    for row in self.data:
      tokenized = self.tokenizer(row[self.feature_col], padding='max_length', truncation=True, max_length=self.max_length)
      tokenized['label'] = torch.tensor(row[self.label_col])
      yield  tokenized 

  def __iter__(self):
    return self.tokenize_stream()
      
         
class TransformerModel(mlflow.pyfunc.PythonModel):
  def __init__(self, tokenizer, model):
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    self.model = AutoModelForSequenceClassification.from_pretrained(model)


  def predict(self, context, model_input):
    """From docs: model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list
    """

    # Convert model input df to list of inputs
    feature_col = model_input.columns[0]
    input_to_lst = model_input[feature_col].to_list()
    tokenized = self.tokenizer(input_to_lst, padding='max_length', truncation=True, max_length=10)

    logits = self.model(torch.tensor(tokenized['input_ids']), 
                        torch.tensor(tokenized['attention_mask'])).logits

    softmax = torch.nn.Softmax(dim=1)
    probs = softmax(logits)
    probs = probs[:,1].detach().numpy()
    # Results are float32 type by default. When applying the model as a Spark UDF, the rounding logic
    # below does is not displayed. Instead, a very long decimal number is returned. This is fixed when
    # conveting the array to double type.
    probs = probs.astype('double')
    probs = np.around(probs, decimals=4)

    return probs
      
      