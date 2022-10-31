# Librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import string
import re
from kmodes.kmodes import KModes
from sklearn import preprocessing
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import math
import io
import torch
from operator import index
from sklearn.metrics.pairwise import euclidean_distances
import csv
import joblib

# Cargar datos 
allItemsDF_o = pd.read_csv("new_items_dataset-v202205.csv", encoding='latin-1')
newItemsDF_o = pd.read_csv("new_items_dataset-v202205_subset.csv", encoding='latin-1') 

# Unir DF
totalItemsDF_o = allItemsDF_o.append(newItemsDF_o, ignore_index=True)

# Función para limpieza de datos
def clean_data(df):
    """
    Esta función realiza la limpieza, imputación y 
    redefinición de las variables. 

    Parametros: Dataframe que se va a modificar
    Return: Dataframe con las variables modificadas
    """
    # Eliminar col 
    df = df.drop(
        columns = ["sub_status", "tags", "seller_country", "buying_mode", 
                "shipping_is_free", "status", "date_created"])

    # Imputación de datos
    df.warranty = df.warranty.fillna('0')

    # Eliminar nulos
    df = df.dropna().reset_index()

    # definir datatypes y variables
    ## numerico
    list_num = pd.Series([
        "price",
        "initial_quantity",	
        "sold_quantity",	
        "available_quantity"
    ])
    for col in list_num:
        df[col] = df[col].astype('float64')

    ## categoria
    list_cat = pd.Series([
        "category_id",	
        "seller_province",	
        "seller_city",	
        "seller_loyalty",	
        "shipping_mode",
        "is_new"
    ])
    for col in list_cat:
        df[col] = df[col].astype('category')

    ## texto
    list_txt = pd.Series([
        "id",
        "title",
        "warranty",
        "seller_id"	,
        "shipping_admits_pickup"
    ])
    for col in list_txt:
        df[col] = df[col].astype('string')

    # Redefinir variables
    ## is_available
    df["available_quantity2"] = df.initial_quantity - df.sold_quantity
    df["is_available"] = np.where(df.available_quantity2 > 0, 1, 0)

    ## sold_amount 
    df["sold_amount"] = df.sold_quantity * df.price
    df = df.drop(columns=["base_price", "initial_quantity", "available_quantity", "available_quantity2"])

    ## seller_province (Capital Federal, Buenos Aires, Otras)
    df["seller_province"] = np.where(df["seller_province"] == "Capital Federal", "CAPITAL FEDERAL", 
    np.where(df["seller_province"] == "Buenos Aires", "BUENOS AIRES", "Otras"))

    ## seller_city
    df["seller_city"] = df.seller_city.str.upper().str.translate(str.maketrans('', '', string.punctuation))

    ## shipping_mode
    df["shipping_mode_me"] = np.where(
        df.shipping_mode == "me1", 1, np.where(df.shipping_mode == "me2", 1, 0))
    df = df.drop(columns = ["shipping_mode"])

    ## shipping_admits_pickup
    df.shipping_admits_pickup = np.where(df.shipping_admits_pickup.str.upper() == "TRUE",1,0)

    ## is_new
    df.is_new = df.is_new.astype('int')

    ## title 
    df.title = df.title.str.lower().str.translate(str.maketrans('', '', string.punctuation))

    ## warranty
    df["warranty"] = df.warranty.str.lower().str.translate(str.maketrans('', '', string.punctuation))
    df["warranty2"] = df.warranty.str.extract('([0-9]+ aã±o)', expand=True)
    df["warranty3"] = df.warranty.str.extract('([0-9]+ mes)', expand=True)
    df["warranty3"] = df.warranty3.fillna(df.warranty2)
    df.warranty = np.where(df.warranty3.isna() == True, 0, 1)
    df = df.drop(columns=["warranty2", "warranty3", "index"])

    # Var cat to dummy
    df = pd.get_dummies(df,columns=["seller_province", "seller_loyalty"])

    return df

totalItemsDF = clean_data(totalItemsDF_o)

# Función para asignar clusters
def kmodes_cluster(df, model):
    """
    Carga el modelo K-modes y asigna a cada elemnto el cluster al que pertence
    Parámetros: DF que contiene los elementos y nombre del archivo joblib
    que contiene el modelo previamente entrenado 
    Return: DF con columna nueva de clusters
    """

    df_items_cat = df[["category_id", "seller_id", "seller_city"]]
    model = joblib.load(model)
    clusters = model.fit_predict(df)
    clustersDF = pd.DataFrame(clusters)
    clustersDF.columns = ['cluster_predicted']
    combinedDF = pd.concat([df_items_cat, clustersDF], axis = 1).reset_index()
    combinedDF = combinedDF.drop(["index"], axis = 1)
    predictionDF = df.drop(columns=["category_id", "seller_id", "seller_city"])
    predictionDF["cluster_predicted"] = combinedDF.cluster_predicted

    return predictionDF

totalItemsDF = kmodes_cluster(totalItemsDF, "model_kmodes.joblib")

# Función para normalizar variables
def mm_norm(col, df):
    """
    Caping de outliers en el percentil 99
    Normaliza una columna usando el metodo min/max
    Parámetros: columna a normalizar (col), Dataframe
    Return: Columna modificada del dataframe
    """

    pd.set_option('mode.chained_assignment', None)
    # caping outliers
    df2 = df[[col]]
    top = df2[col].quantile(0.99)
    df2[col] = np.where(df2[col] > top, top, df2[col])

    # normalization
    x = df2.values 
    transformer = preprocessing.MinMaxScaler()
    x_scaled = transformer.fit_transform(x)
    DFnorm = pd.DataFrame(x_scaled, columns=[col])
    return DFnorm

totalItemsDF["price"] = mm_norm("price", totalItemsDF)
totalItemsDF["sold_amount"] = mm_norm("sold_amount", totalItemsDF)
totalItemsDF["sold_quantity"] = mm_norm("sold_quantity", totalItemsDF)

# separar DF
newItemsDF = totalItemsDF.tail(1)
allItemsDF = totalItemsDF.loc[~totalItemsDF['id'].isin(newItemsDF.id)]

# Función para calcular distancias
def distance(point_x, point_y, n):
    """
    Calcula la distancia euclidiana entre los elementos
    del conjuto x y el conjunto y
    Parámetros: 
        - point_x: DF elementos x
        - point_y: DF elemento y
        - n: número de elementos semenjantes 
            a seleccionar
    Return:
        - similarDF: DF de elementos semejantes
        - idSimilar: id de los elementos 
    """
    # convertir datos a vectores
    pointXvector = point_x.drop(columns=["id", "title"])
    pointXvector = pointXvector.values
    pointYvector = point_y.drop(columns=["id", "title"])
    pointYvector = pointYvector.values 

    # calcular distancia euclidiana entre cada punto 
    dist = euclidean_distances(pointXvector, pointYvector)

    # recuperar id de los elementos
    dist = pd.DataFrame(dist, columns = ["distance"])
    pointXDF = point_x[["id", "title"]]
    pointYDF = point_y[["id", "title"]]
    pointXDF["distance"] = dist.distance
    pointXDF["id_new_item"] = pointYDF.id.max()
    pointXDF["title_new_item"] = pointYDF.title.max()
    similarDF = pointXDF.sort_values(by="distance").head(n)
    # set de elementos semejantes
    idSimilar = similarDF.id

    return similarDF, idSimilar

similarDF, idSimilar = distance(allItemsDF, newItemsDF, 100)

# Sentence similarity

def sentence_similarity(newItemsDF, similarDF, train_model, tensor_emb, n):
    
    """
    Calcular la similitud entre dos frases/parrafos
    usando el modelo pre-entrenado 
    Parametros: 
        - newItemsDF: DF que contiene las frases
        - similarDF: DF de elementos cercanos determinado por 
        medio de la medida de distancias
        - train_model: modelo 
        - tensor_emb: tensor de caracteristicas de cada
        elemento de texto del DF newItemsDF
        - n: cantidad de elementos semenjantes a encontrar
    Return:
        - similarDFtop: Descripcion de elmentos similares
        - idSimilarTop: id de elementos similares
    """

    # cargar modelo
    model = SentenceTransformer(train_model)

    # cargar tensor de caracteristicas
    embeddingsAll = torch.load(tensor_emb)

    # filtrar tensor 
    indices = torch.tensor([similarDF.index.values])
    embeddingsFiltered = torch.index_select(embeddingsAll, 0, indices.squeeze())

    # reiniciar index de similares
    similarDF = similarDF.reset_index()

    # Solo titulos
    newItems = list(newItemsDF.title)
    newItems = newItems * len(similarDF)

    # Embedding
    embeddings1 = model.encode(newItems, convert_to_tensor=True)

    # Cosine-similarities
    cosineScores = util.cos_sim(embeddings1, embeddingsFiltered)
    indices = torch.tensor([0])
    cosineScores = torch.index_select(cosineScores, 0, indices)
    cosineScores = pd.DataFrame(cosineScores.numpy(), index=["sentence_similarity"]).T

    # unir al dataframe inicial 
    similarDF["sentence_similarity"] = cosineScores

    # top más similares
    similarDFtop = similarDF.sort_values(by="sentence_similarity", ascending=False)
    similarDFtop = similarDFtop.head(n)
    idSimilarTop = similarDFtop.id
    return similarDFtop, idSimilarTop

similarDFtop, idSimilarTop = sentence_similarity(
    newItemsDF, similarDF, 'sentence_similarity_spanish_es', 
    'tensor_embeddingsAll.pt', 5)

# guardar resultado en excel
similarDFtop.to_csv('new_item_similarities.csv')
