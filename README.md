# COMP 7970: Natural Language Processing
**Fall 2021 / Auburn University**

This repository is my implementation of assignments and projects of COMP 7970: Natural Language Processing. 

## Assignment 1: Latent Dirichlet Allocation Implementation

✅ Q1: Implementation of LDA

✅ Q2: Visualization Of Topics

✅ Q3: Implement KL-divergence

## Assignment 2: Implementation of Word2Vec

In this assignment, I've implemented a custom Word2Vec model following the paper **Efficient Estimation of Word Representation in Vector Space**. I've implemented the Skip-gram model using **PyTorch** where word embedding size is 50. The model has been trained on [Amazon Fine Food Review Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews) upto 10 epochs and the checkpoint file can be found [here](https://github.com/Mousumi44/Natural-Language-Processing/blob/main/assignment%202/my_checkpoint.pth.tar)

✅ Q1: Implementation of Word2Vec

✅ Q2: Find Similar Words by loading the saved model

* Coffee : craving 
* Tuna: sudorific

✅ Q3: Word Analogies with GloVe

In this part, I've used GloVe 300d vector and tested different word analogies. For example:
* Spain is to Spanish as Germany is *togerman*
* Japan is to Tokyo as France is to *paris*

## Assignment 3: Implementation of Text Summarization

In this assignment, I've implemented two differenttext summarizer model using 1) [Google PEGASUS](https://huggingface.co/transformers/model_doc/pegasus.html) and 2) [Facebook BART](https://huggingface.co/transformers/model_doc/bart.html). I used the pretrained model from huggingface and later fine-tuned on CNN/DailyMail dataset.


✅ Q1: Implementation of Text Summarizer

|  Model  |         Script        | URL for Fine-tuned Model on huggingface |
|:-------:|:---------------------:|:---------------------------------------:|
| Pegasus | [Pegasus python Script](https://github.com/Mousumi44/Natural-Language-Processing/blob/main/assignment3/pegasus_finetune_cnn.py) | [https://huggingface.co/Mousumi/finetuned_pegasus](https://huggingface.co/Mousumi/finetuned_pegasus)                                       |
|   BART  |   [BART python Script](https://github.com/Mousumi44/Natural-Language-Processing/blob/main/assignment3/bart_finetune_cnn.py)  | [https://huggingface.co/Mousumi/finetuned_bart](https://huggingface.co/Mousumi/finetuned_bart)                                     |

✅ Q2: Evaluation of Text Summarizer 

### To Run the [test.py](https://github.com/Mousumi44/Natural-Language-Processing/blob/main/assignment3/test.py)

#### For Pegasus:

```
python test.py --model=pegasus
```

#### For BART:

```
python test.py --model=bart
```

#### Results:

|  Model  |           | ROUGE-1 |       |           | ROUGE-2 |       |           | ROUGE-3 |      |           | ROUGE-L |       |
|:-------:|:---------:|:-------:|:-----:|:---------:|:-------:|:-----:|:---------:|:-------:|:----:|:---------:|:-------:|:-----:|
|         | Precision |  Recall |   F1  | Precision |  Recall |   F1  | Precision |  Recall |  F1  | Precision |  Recall |   F1  |
| Pegasus |   28.36   |  50.72  |  34.1 |   11.74   |  20.55  |  14.0 |    6.63   |  11.41  | 7.79 |   18.37   |  33.83  | 22.25 |
|   BART  |   56.51   |   15.4  | 23.64 |   26.55   |   6.77  | 10.52 |   14.93   |   3.58  | 5.61 |    46.1   |  12.56  | 19.27 |

#### References:

[1. Model sharing and uploading huggingface](https://huggingface.co/transformers/model_sharing.html)


