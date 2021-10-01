# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:50:04 2021

@author: pouph
"""

import pandas as pd
import spacy
nlp = spacy.load("fr_core_news_sm")

# Lecture du document
data = pd.read_excel('pressesArticle.xlsx')
# Voire quelque ligne du document
print(data.head())
# Pour lire un article complet:
#print(data['Article'][0])
# la taille de document
#print(len(data))

"""
PREPROCESSING: Dans ce cas nous allons faire une preparation de donnees elementaire
avec deux parametres importants de la fonction TfidfVectorizer qui sont:
    max_df: float in range [0.0, 1.0] or int, default=1.0
        Lors de la construction des vocabulaires,  il permet d'ignorer les termes dont la fréquence du document est strictement supérieure au seuil donné (mots d’arrêt spécifiques au corpus). 
        Si max_df est  un float, le paramètre représente une proportion de tous les mots dans le documents, 
        Si max_df est entier, c'est un nombre absolus du nombre de mots dans le document. 
        Ce paramètre est ignoré si le vocabulaire n’est pas Aucun.
        
    min_df: min_df: float in range [0.0, 1.0] or int, default=1
        Lors de la construction des vocabulaires,  il permet d'ignorer les termes dont la fréquence du document est strictement inferieure au seuil donné. 
        Cette valeur est aussi appelee cut-off dans la litterature
        Si max_df est  un float, le paramètre représente une proportion de tous les mots dans le documents, 
        Si max_df est entier, c'est un nombre absolus du nombre de mots dans le document. 
        Ce paramètre est ignoré si le vocabulaire n’est pas Aucun.
    
    stop_words: {‘english’}, list, default=None

Pour plus de detailles: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Cette fonction va creer une matrice de grande taille donc la dimension est egale nobre d'article * le nombre total de tous les mots(mots unique) dans le text' 
"""
"""
Visualisation des tops mots dans chaque document ou article
"""
# Fonction de visualisation
from nltk.corpus import stopwords
from time import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

spacy_stop_words = [stop for stop in nlp.vocab]
final_stopwords_list = stopwords.words('french') + spacy_stop_words
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words = final_stopwords_list)
tf_id = tf_vectorizer.fit_transform(data['Article'])
#print(type(tf_id))

## LDA
n_components =5
random_state = 42
model_LDA = LatentDirichletAllocation(n_components = n_components, random_state = random_state)

# voire les parametres du modele
print(model_LDA.fit(tf_id))
print('**********************************************', end='\n')

# Showing Stored Words
len(tf_vectorizer.get_feature_names())

# Imprimer quelque feature names
#import random
feature_names = []
for i in range(len(tf_vectorizer.get_feature_names())):
    word_id = i #random.randint(0,54776)
    feature_names.append(tf_vectorizer.get_feature_names()[word_id])
    #print(tf_vectirizer.get_feature_names()[word_id])
print('**********************************************', end='\n')

# afficher les features names.
#print(feature_names)

#https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

n_top_words = 5

plot_top_words(model_LDA, feature_names, n_top_words, 'Topics in LDA model')

"""
Dans cette section nous allons ecrire une function pour exploirer un topic precis
"""
def top_words_single_topic(topic = 0, n = 10):
    single_topic = model_LDA.components_[topic]
    top_word_indeces = single_topic.argsort()[-n:]
    for index in top_word_indeces:
        print(tf_vectorizer.get_feature_names()[index])
        
# Exemple 
n =5
topic = 0
top_words_single_topic(topic,n)
print('**********************************************', end='\n')

"""
Fonction pour afficher les tops n mots pour different Topic
"""
def top_words_each_topic(n):
    for index,topic in enumerate(model_LDA.components_):
        print(f'The top #{n} words for topic #{index}')
        print([tf_vectorizer.get_feature_names()[i] for i in topic.argsort()[-n:]])
        print('\n')

# Exemple
n = 5
top_words_each_topic(n)
print('**********************************************', end='\n')



