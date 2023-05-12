#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""This module demonstrates how to create an ngram model
using NLTK and it applies to to predict punctuation."""

from nltk.tokenize import wordpunct_tokenize  # a simple regex tokenizer
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

documents = [
    "Franz jagt im komplett verwahrlosten Taxi quer durch Bayern.",
    "Franz, ist das Verwahloste da dein Taxi?",
    "Ich mag Taxis, Bayern und Franz, wenn Franz in Bayern ist.",
    "Franz' Taxi ist verwahlost, weil es in Bayern ist.",
    "Bayern ist verwahlost, weil Franz in Bayern ist."
    "Franz ist das Taxi, das Franz in Bayern jagt."
]

tokenized = [wordpunct_tokenize(sentence) for sentence in documents]
# tokenized = [["Franz", "jagt", "im", "komplett", ..., "Bayern", "."]]

n = 3
train, vocab = padded_everygram_pipeline(n, tokenized)  # do NOT concatenate beforehand, preserve sentences

lm = MLE(3)
lm.fit(train, vocab)

print(lm.score("Franz"))
print(lm.score("Bayern", ["Franz", "in"]))
print(lm.score("weil", ["ist", "verwahrlost"]))

test = [(",", "weil", "das"), ("Franziska", "ist", "ein"), ("Franz", ",", "Taxis")]

print(lm.entropy(test))  # average amount of surprise
print(lm.perplexity(test))  # perplexity is low (good) if the probabilities are high, but this depends on vocab size

print(lm.generate(5, random_seed=42))
print(lm.generate(3, random_seed=13))
print(lm.generate(8, random_seed=7))

print(lm.generate(4, text_seed=["<s>", "Franz"], random_seed=42))
print(lm.generate(2, text_seed=["ist", "."], random_seed=13))

def predict_commas(sentence: str) -> list:
    tokenized_sentence = wordpunct_tokenize(sentence)
    tokenized_sentence.insert(0, "<s>")
    tokenized_sentence.append(r"<\s>")
    output = tokenized_sentence

    for i, _ in enumerate(tokenized_sentence[1:-2]):  # not interested in commas before first or after 2nd-to-last word
        comma_score = lm.score(",", [tokenized_sentence[i-1], tokenized_sentence[i]])
        next_token_score = lm.score(tokenized_sentence[i+1], [tokenized_sentence[i-1], tokenized_sentence[i]])
        if comma_score > next_token_score:
            output.insert(i+1, ",")
    return output

prediction = predict_commas("Franz Taxis und Bayern sind verwahrlost weil Franz in Bayern ist.")
print(" ".join(prediction))
