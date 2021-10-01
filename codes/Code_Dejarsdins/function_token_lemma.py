# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:03:09 2021

@author: pouph
"""

"""
Fonction pour afficher les tockens, lemma
"""
# Telechargement des packages
import spacy
from spacy import displacy
# Telechrgement du modele  et NPL_object en Francais
nlp = spacy.load('fr_core_news_sm')

#Token an Lemma
def show_lemmas(text):
    for token in text:
        print(f'{token.text:{12}} {token.pos_:{6}} {token.lemma:<{22}} {token.lemma_}')

#Here we're using an f-string to format the printed text by setting minimum field widths and adding a left-align to the lemma hash value.

# Exemple
doc = nlp(u"Poupheulie Gabriel a commencé son stage à Desjardin le 3 août 2012. Michelk et Jacqueline sont étudiant au HEC Montréal. Ce stage sera considéré comme son projet supervisé. Son encadreur universitaire est Gilles Caporossi et celui de Desjardins est Laurent. En cas d'urgence , il faut appeler son épouse Kambou Kam Yolande-Viviane au 514-572-6082. Ce stage est financé par le programme Mitacs a hauteur de 15 000 $CAD. Yolanded, jmichel ont pour adresse papa@hotmail.com")
show_lemmas(doc)

#Named Entity Recognition (NER)
# Write a function to display basic entity info:
def show_ents(text):
    if text.ents:
        for ent in text.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')
        
#Example
show_ents(doc)
"""
# Funtion to give the morphologie of the token or NER

def show_morph(text):
   for token in text:
        if((token.pos_ == 'NOUN') | (token.pos_ == 'PRON')):
           print(f'{token.text:{12}} {token.morph}')

#Example
show_morph(doc)
"""

"""
for sent in doc2.sents:
    docx = nlp(sent.text)
    if docx.ents:
        displacy.render(docx, style='ent', jupyter=True)
    else:
        print(docx.text)
Visualizing Named Entities¶
Besides viewing Part of Speech dependencies with style='dep', displaCy offers a style='ent' visualizer:
"""
# function to disply NER
def disply_ents(text):
    sentence_spans = list(text.sents)
    displacy.serve(sentence_spans, style='ent')
    
#Exemple
#disply_ents(doc)

# Fonction qu colorie juste quelque type de ENR
type_ents = ['PERSON','NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT','EVENT','WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME','PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
# on peut definir les couleur pour noirsir les NER
ent_colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'PRODUCT': 'radial-gradient(yellow, green)', 'PERSON': 'black', 'PER': 'black'}

def disply_some_ents(text, ents = type_ents, colors=ent_colors):
    options = {'ent':type_ents, 'colors':colors}
    displacy.serve(text, style='ent', options = options)
# Exemple
disply_some_ents(doc, ents = type_ents, colors=ent_colors)









