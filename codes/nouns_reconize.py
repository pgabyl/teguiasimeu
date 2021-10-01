# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 13:15:26 2021

@author: pouph
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
# Initialisation d'un fichier vide
data = []
# Ouvrir le text, lire chaque ligne et ajouter dans data en evitant les ligne vide.
with open('./testFille.txt',"r")as myfile:
    for line in myfile:
        line = line.replace('\n', '')
        if line == '':
            continue
        else:
            data.append(str(line))
myfile.close()

import spacy
from spacy.gold import GoldParse, Doc
from spacy.vocab import Vocab
from collections import Counter
