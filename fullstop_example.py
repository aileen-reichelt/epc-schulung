#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""This module tests the usage and performance of a BERT model
pretrained for punctuation restoration, "FullStop".
GitHub:
https://github.com/oliverguhr/fullstop-deep-punctuation-prediction
Huggingface:
https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large
"""

from deepmultilingualpunctuation import PunctuationModel

model = PunctuationModel()
text = "Mein Name ist Klara 18 und ich lebe in Heidelberg Baden-Württemberg Deutschland Ist das eine Frage Frau Müller"

clean_text = model.preprocess(text)
print(clean_text)

labeled_words = model.predict(clean_text)
print(labeled_words)

result = model.restore_punctuation(text)
print(result)