import torch
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class BartDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])  # torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])  # len(self.labels)

def prepare_data(model_name, 
    train_texts, train_labels, 
    val_texts=None, val_labels=None, 
    test_texts=None, test_labels=None):
    """
    Prepare input data for model fine-tuning
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)

    prepare_val = False if val_texts is None or val_labels is None else True
    prepare_test = False if test_texts is None or test_labels is None else True

    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation=True, padding=True)
        decodings = tokenizer(labels, truncation=True, padding=True)
        dataset_tokenized = BartDataset(encodings, decodings)
        return dataset_tokenized

    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels) if prepare_val else None
    test_dataset = tokenize_data(test_texts, test_labels) if prepare_test else None

    return train_dataset, val_dataset, test_dataset, tokenizer

def prepare_fine_tuning(model_name, tokenizer, train_dataset, val_dataset=None, freeze_encoder=False, output_dir='./bart_results'):
  """
  Prepare configurations and base model for fine-tuning
  """
  torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = BartForConditionalGeneration.from_pretrained(model_name).to(torch_device)

  if freeze_encoder:
    for param in model.model.encoder.parameters():
      param.requires_grad = False

  if val_dataset is not None:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2,             #total number of training epochs
      per_device_train_batch_size=1,  # batch size per device during training, can increase if memory allows
      per_device_eval_batch_size=1,   # batch size for evaluation, can increase if memory allows
      save_steps=5,                    # number of updates steps before checkpoint saves
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      evaluation_strategy='steps',     # evaluation strategy to adopt during training
      eval_steps=2,                    # number of update steps before evaluation
      warmup_steps=5,                  # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./bart_logs',            # directory for storing logs
      logging_steps=2,
      gradient_accumulation_steps = 256
    )

    trainer = Trainer(
      model=model,                         # the instantiated ???? Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      eval_dataset=val_dataset,            # evaluation dataset
      tokenizer=tokenizer
    )

  else:
    training_args = TrainingArguments(
      output_dir=output_dir,           # output directory
      num_train_epochs=2,            # total number of training epochs
      per_device_train_batch_size=1,   # batch size per device during training, can increase if memory allows
      save_steps=5,                  # number of updates steps before checkpoint saves
      save_total_limit=1,              # limit the total amount of checkpoints and deletes the older checkpoints
      warmup_steps=5,                # number of warmup steps for learning rate scheduler
      weight_decay=0.01,               # strength of weight decay
      logging_dir='./bart_logs',            # directory for storing logs
      logging_steps=2,
      gradient_accumulation_steps = 256
    )

    trainer = Trainer(
      model=model,                         # the instantiated ???? Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=train_dataset,         # training dataset
      tokenizer=tokenizer
    )

  return trainer

if __name__=='__main__':
    # use CNN/DM dataset as example, with first 1000 docs as training data
    dataset = load_dataset("cnn_dailymail", '3.0.0')
    train_texts, train_labels = dataset['train']['article'][:10000], dataset['train']['highlights'][:10000]
    valid_texts, valid_labels = dataset['validation']['article'][:500], dataset['validation']['highlights'][:500]
    test_texts, test_labels = dataset['test']['article'][:500], dataset['test']['highlights'][:500]

    
    
    # use BART  Base model as base for fine-tuning
    # use BART Base model as base for fine-tuning
    model_name = 'facebook/bart-base'
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts = valid_texts, val_labels=valid_labels,
                test_texts=test_texts, test_labels=test_labels)

    

    trainer = prepare_fine_tuning(model_name, tokenizer, train_dataset)
    trainer.train()
    trainer.evaluate(test_dataset)

    # sys.exit()

