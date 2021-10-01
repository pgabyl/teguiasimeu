# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:35:23 2021
Master Thesis
@author: pouph
"""

# Telechargement des packages
import spacy
# Telechrgement du modele  et NPL_object en Francais
nlp = spacy.load('fr_core_news_sm')
#print(dir(nlp))

# Create a Doc object
doc = nlp(u"Poupheulie Gabriel fait son stage à la fondation Desjardin")
#doc = nlp(u"Poupheulie Gabriel a commencé son stage à Desjardin le 3 août 2012. Il est étudiant au HEC Montréal. Ce stage sera considéré comme son projet supervisé. Son encadreur universitaire est Gilles Caporossi et celui de Desjardins est Laurent. En cas d'urgence , il faut appeler son épouse Kambou Kam Yolande-Viviane au 514-572-6082. Ce stage est financé par le programme Mitacs a hauteur de 15 000 $CAD.")
# Imprimer chaque token du document.
for token in doc:    
    print(token.text, token.pos_, token.dep_,)
    # Juste prendre le token de type alpha numerique
    #if(token.is_alpha != False):
        #print(token.text, token.pos_, token.dep_,)
    #if(token.pos_ == 'PROPN'):
        #print(token.text, token.pos_, token.dep_,)

# Named Entities
"""
Going a step beyond tokens, named entities add another 
layer of context. The language model recognizes 
that certain words are organizational names 
while others are locations, and still other combinations relate 
to money, dates, etc. Named entities are accessible through 
the ents property of a Doc object.
"""
for ent in doc.ents:
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    # reconnaitre juste les noms des personne
    #if(ent.label_ == "LOC"):
       #print(ent.text) 
"""
Noun Chunks
"""
for chunk in doc.noun_chunks:
    print(chunk.text, chunk.label_)

"""
Built-in Visualizers
spaCy includes a built-in visualization tool called displaCy. 
displaCy is able to detect whether you're working in a Jupyter notebook, 
and will return markup that can be rendered in a cell right away. 
When you export your notebook, the visualizations will be included as HTML.

For more info visit https://spacy.io/usage/visualizers
"""
from spacy import displacy
"""
Visualizing long texts
Long texts can become difficult to read when displayed in one row, 
so it’s often better to visualize them sentence-by-sentence instead.
 As of v2.0.12, displacy supports rendering both Doc 
 and Span objects, as well as lists of Docs or Spans. 
 Instead of passing the full Doc to displacy.serve, 
 you can also pass in a list doc.sents. This will create one visualization for each sentence.
"""
#doc = nlp(u"Poupheulie Gabriel fait son stage à la fondation Desjardin")
#sentence_of_doc = list(doc.sents)
#displacy.serve(sentence_of_doc, style='ent')