#Se importan las librerias necesaiton
import sys, os
import pandas as pd
import numpy as np
import scipy
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
#SciKit-Learning para dividir el dataset en testing y training
from sklearn.model_selection import train_test_split
#Se usa la funcion pairwise distance (comparacion de distancias)
from sklearn.metrics.pairwise import pairwise_distances
#Se evalua el resultado usando Mean Square Error (MSE)
from sklearn.metrics import mean_squared_error
from math import sqrt

os.environ.setdefault("DJANGO_SETTINGS_MODULE","srswdom.settings") 

import django
django.setup()

from recomendador.models import Calificacion, Servicio, Usuario

#Se hacen predicciones
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        #Usamos np.newaxis para que mean_user_rating Tenga el mismo formato que ratings 
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

# comando: python metodoColaborativo.py 
if __name__ == "__main__":
  if len(sys.argv) == 1:
    calificacion_df = pd.DataFrame(list(Calificacion.objects.all().values()))
    num_cal = Calificacion.objects.count()
    print(calificacion_df.head(5))
    servicio_df = pd.DataFrame(list(Servicio.objects.all().values()))
    num_servicio = Servicio.objects.count()
    print(servicio_df.head(5))
    usuario_df = pd.DataFrame(list(Usuario.objects.all().values()))
    num_usuario = Usuario.objects.count()
    print(usuario_df.head(5))

    #Calculamos las interacciones que existen entre los tipos de usuario y tipos de servicio
    
      
    print("Existen {} Calificaciones ".format(num_cal))
    print("Existen {} Servicios ".format(num_servicio))
    print("Existen {} Usuarios ".format(num_usuario))
  
#Para evitar el problema de inicio en frio, se va a limitar el DS a usarios que tengan al menos 5 calificaciones, puesto
 # [calificacion][fecha_calificacion][id][servicio_id][usuario_id]
 # [user id ][item id ][rating][Timestamp]

#Se separa el Data ser en train y test
    train_data, test_data = train_test_split(calificacion_df,test_size=0.25)
#Crear dos matrices usuari-item, una para entrenamiento y otra para pruebas
    train_data_matrix = np.zeros((num_usuario, num_servicio))
   
    for line in train_data.itertuples():
      train_data_matrix[line[5]-1, line[4]-1] = line[1]

    test_data_matrix = np.zeros((num_usuario, num_servicio))
    for line in test_data.itertuples():
      test_data_matrix[line[5]-1, line[4]-1] = line[1]
    

    user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
    item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
    item_prediction = predict(train_data_matrix, item_similarity, type='item')
    print("itemPrediction: ",item_prediction)
    user_prediction = predict(train_data_matrix, user_similarity, type='user')
    print("userPrediction: ",user_prediction)

    print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
    print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))

  else:
    print("Por favor, introduce la direccion del archivo Usuarios") 





"""
      usuarios_df.apply(
      guardar_usuario_de_fila,# Function to apply to each column/row

      axis=1 #axis : {0 or ‘index’, 1 or ‘columns’}, default 0
          #     0 or ‘index’: apply function to each column
              #1 or ‘columns’: apply function to each row
    )
"""




"""articles_df = pd.read_csv('input/shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)

#print(articles_df.head(5))

interactions_df = pd.read_csv('input/users_interactions.csv')
interactions_df.head(10)
"""
#print(interactions_df.head(10))
"""

PROBABLMENTE ASIGNE UN VALOR A ESTE TIPO DE EVENTO PARA PODER SER CALCULADO, EN NUESTRO CASO NO SE 
NECESITA POR QUE EL VALOR DE LA CALIFICACION YA ES DIRECTA
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])

#Se limita solo a usuario que han realizado 5 interacciones

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

#Se definen interacciones unicas


def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)
#print(interactions_full_df.head(10))

#Evaluacion

#Interacciones entre los conjuntos
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

#Indexing by personId to speed up the searches during evaluation
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')



def get_items_interacted(person_id, interactions_df):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])

#Top-N accuracy metrics consts
EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100

class ModelEvaluator:


    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
            try:
                index = next(i for i, c in enumerate(recommended_items) if c == item_id)
            except:
                index = -1
            hit = int(index in range(0, topn))
            return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df
    
model_evaluator = ModelEvaluator()  

#Computes the most popular items
item_popularity_df = interactions_full_df.groupby('contentId')['eventStrength'].sum().sort_values(ascending=False).reset_index()
item_popularity_df.head(10)

#print(item_popularity_df.head(10))

#POPULARIDAD
#Este metodo realiza una evaluacion del modelo de popularidad, de acerdo a lso metodos descritos. obtiene el Recall@5 de 0.2417
#Quiere decir que el 24% de los intem interactuado en el conjunto de prieba fueron calificado por un modelo popularidad de los top 5 
#items (de una lisra de 100 items aleatorios)
class PopularityRecommender:
    
    MODEL_NAME = 'Popularity'
    
    def __init__(self, popularity_df, items_df=None):
        self.popularity_df = popularity_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Recommend the more popular items that the user hasn't seen yet.
        recommendations_df = self.popularity_df[~self.popularity_df['contentId'].isin(items_to_ignore)] \
                               .sort_values('eventStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['eventStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
#QUITAR ESTE COMENTARIO PARA PROBAR ESTE METODO
"""
"""
popularity_model = PopularityRecommender(item_popularity_df, articles_df)
print('Evaluating Popularity recommendation model...')
pop_global_metrics, pop_detailed_results_df = model_evaluator.evaluate_model(popularity_model)
print('\nGlobal metrics:\n%s' % pop_global_metrics)
pop_detailed_results_df.head(10)
"""
#print(pop_detailed_results_df.head(10))

#FILTRO COLABORATIVO
"""
#Creating a sparse pivot table with users in rows and items in columns
users_items_pivot_matrix_df = interactions_train_df.pivot(index='personId', 
                                                          columns='contentId', 
                                                          values='eventStrength').fillna(0)

users_items_pivot_matrix_df.head(10)
#print(users_items_pivot_matrix_df.head(10))
users_items_pivot_matrix = users_items_pivot_matrix_df.as_matrix()
users_items_pivot_matrix[:10]
#print(users_items_pivot_matrix[:10])

users_ids = list(users_items_pivot_matrix_df.index)
users_ids[:10]
#print(users_ids[:10])



#The number of factors to factor the user-item matrix.
NUMBER_OF_FACTORS_MF = 15
#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)

U.shape
#print(U.shape)
Vt.shape
#print(Vt.shape)

sigma = np.diag(sigma)
sigma.shape
#print(sigma.shape)

#After the factorization, we try to to reconstruct the original matrix by multiplying its factors. The resulting matrix is not sparse any more. 
#It was generated predictions for items the user have not yet interaction, which we will exploit for recommendations.
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings
#print(all_user_predicted_ratings)

#Converting the reconstructed matrix back to a Pandas dataframe
cf_preds_df = pd.DataFrame(all_user_predicted_ratings, columns = users_items_pivot_matrix_df.columns, index=users_ids).transpose()
cf_preds_df.head(10)
#print(cf_preds_df.head(10))
len(cf_preds_df.columns)
#print(len(cf_preds_df.columns))

#Evaluating the Collaborative Filtering model (SVD matrix factorization), 
#we observe that we got Recall@5 (33%) and Recall@10 (46%) values higher than Popularity model, but lower than Content-Based model.
#It appears that for this dataset, Content-Based approach is being benefited by the rich item attributes (text) for a better modeling of users preferences.

class CFRecommender:
    
    MODEL_NAME = 'Collaborative Filtering'
    
    def __init__(self, cf_predictions_df, items_df=None):
        self.cf_predictions_df = cf_predictions_df
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.cf_predictions_df[user_id].sort_values(ascending=False) \
                                    .reset_index().rename(columns={user_id: 'recStrength'})

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = sorted_user_predictions[~sorted_user_predictions['contentId'].isin(items_to_ignore)] \
                               .sort_values('recStrength', ascending = False) \
                               .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]


        return recommendations_df
    
cf_recommender_model = CFRecommender(cf_preds_df, articles_df)

print('Evaluating Collaborative Filtering (SVD Matrix Factorization) model...')
cf_global_metrics, cf_detailed_results_df = model_evaluator.evaluate_model(cf_recommender_model)
print('\nGlobal metrics:\n%s' % cf_global_metrics)
print(cf_detailed_results_df.head(10))

"""