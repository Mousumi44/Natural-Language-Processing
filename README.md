# COMP 7970: Natural Language Processing
**Fall 2021 / Auburn University**

This repository is my implementation of assignments and projects of COMP 7970: Natural Language Processing. 

## Assignment 1: Latent Dirichlet Allocation Implementation

✅ Q1: Implementation of LDA

✅ Q2: Visualization Of Topics

✅ Q3: Implement KL-divergence

## Assignment 2: Implementation of Word2Vec

In this assignment, I've implemented a custom Word2Vec model following the paper ``Efficient Estimation of Word Representation in Vector Space''. I've implemented the Skip-gram model using **PyTorch** where word embedding size is 50. The model has been trained on [Amazon Fine Food Review Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews) upto 10 epochs and the checkpoint file can be found [here](https://github.com/Mousumi44/Natural-Language-Processing/blob/main/assignment%202/my_checkpoint.pth.tar)

✅ Q1: Implementation of Word2Vec

✅ Q2: Find Similar Words

* Coffee : craving items quality stores mints
* Tuna: sudorific

✅ Q3: Word Analogies with GloVe

In this part, I've used GloVe 300d vector and tested different word analogies. For example:
* Spain is to Spanish as Germany is *togerman*
* Japan is to Tokyo as France is to *paris*
