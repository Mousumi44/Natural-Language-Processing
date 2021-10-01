from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys
import pandas as pd
import re
from datasets import load_metric
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description='Process argument')
parser.add_argument('--model', default='bart', help='which correlation metric to use')
args = parser.parse_args()


regex = re.compile('[.,@_!#$%^&*()<>?/\|}{~:]')

def rougeScore(hyp, ref):
  metric = load_metric('rouge')
  metric.add_batch(predictions=hyp, references=ref)
  score = metric.compute(rouge_types=["rouge1", "rouge2", "rouge3", "rougeL", "rougeLsum"],
                                        use_agregator=True, use_stemmer=True)
  return score

def parse_score(result):
    ## returns percentage f1 scroe (mid value)
    ## other options are low and high

    #For all precision, recall, f1 inspection
    scoreFinal = []
    for k,v in result.items():
          listScore = []
          listScore.append(k)
          listScore.append(round(v.mid.precision * 100, 2))
          listScore.append(round(v.mid.recall * 100, 2))
          listScore.append(round(v.mid.fmeasure * 100, 2))
          scoreFinal.append(listScore)
    return scoreFinal   
    # return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

def preprocess(text):
    text = text.replace("\'\'", '"')
    text = ' '.join(text.split())
    sents = text.replace('""', '"').split("<n>")
    sents = [sent for sent in sents if sent.strip() != ""]
    out = [sent+'.' if regex.search(sent.strip()[-1]) == None else sent for sent in sents]
    return " ".join(out)


test_doc = []
test_highlights = []

def helper(test_file='cnn_dailymail_test_assignmemt3.csv'):
    df = pd.read_csv(test_file)

    for i in range(len(df)):
        test_doc.append(preprocess(df['article'][i]))
        test_highlights.append(preprocess(df['highlights'][i]))
    return test_doc, test_highlights 


#Prapare Dataset
helper()
print(len(test_doc))
print(len(test_highlights))

'''
src_text = [" PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow.",
"In the end, it played out like a movie. A tense, heartbreaking story, and then a surprise twist at the end. As eight of Mary Jane Veloso's fellow death row inmates -- mostly foreigners, like her -- were put to death by firing squad early Wednesday in a wooded grove on the Indonesian island of Nusa Kambangan, the Filipina maid and mother of two was spared, at least for now. Her family was returning from what they thought was their final visit to the prison on so-called \"execution island\" when a Philippine TV crew flagged their bus down to tell them of the decision to postpone her execution. Her ecstatic mother, Celia Veloso, told CNN: \"We are so happy, so happy. I thought I had lost my daughter already but God is so good. Thank you to everyone who helped us."]
'''

src_text = test_doc
result = []
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model =="bart":
    tokenizer = AutoTokenizer.from_pretrained("Mousumi/finetuned_bart")

    model = AutoModelForSeq2SeqLM.from_pretrained("Mousumi/finetuned_bart").to(torch_device)

elif args.model=="pegasus":
    tokenizer = AutoTokenizer.from_pretrained("Mousumi/finetuned_pegasus")
    model = AutoModelForSeq2SeqLM.from_pretrained("Mousumi/finetuned_pegasus").to(torch_device)

else:
    print("Provide Correct Model Name")
    sys.exit()

no_samples = len(src_text)

for i in range(no_samples):
    with tokenizer.as_target_tokenizer():
        tokenized_text = tokenizer([src_text[i]], return_tensors='pt', padding=True, truncation=True)
    batch = tokenized_text.to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    result.append(tgt_text[0])

score = rougeScore(result[:no_samples],test_highlights[:no_samples])
print(parse_score(score))
