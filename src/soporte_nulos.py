# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

import math 
# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

# Visualización de datos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns


class GestionNulos:
    """
    Clase para gestionar los valores nulos en un DataFrame.
    """

    def __init__(self, dataframe):
        """
        Inicializa la clase con un DataFrame.

        Parámetros:
        - dataframe: DataFrame de pandas.
        """
        self.dataframe = dataframe
    
    def calcular_porcentaje_nulos(self):
        """
        Calcula el porcentaje de valores nulos en el DataFrame.

        Retorna:
        - Series: Porcentaje de valores nulos para cada columna con valores nulos.
        """
        df_nulos = (self.dataframe.isnull().sum() / self.dataframe.shape[0]) * 100
        return df_nulos[df_nulos > 0]
    
    def seleccionar_columnas_nulas(self):
        """
        Selecciona las columnas con valores nulos.

        Retorna:
        - Tuple: Tupla de dos elementos con las columnas categóricas y numéricas que tienen valores nulos.
        """
        nulos_esta_cat = self.dataframe[self.dataframe.columns[self.dataframe.isnull().any()]].select_dtypes(include="O").columns
        nulos_esta_num = self.dataframe[self.dataframe.columns[self.dataframe.isnull().any()]].select_dtypes(include=np.number).columns
        return nulos_esta_cat, nulos_esta_num

    def mostrar_distribucion_categoricas(self):
        """
        Muestra la distribución de categorías para las columnas categóricas con valores nulos.
        """
        col_categoricas = self.seleccionar_columnas_nulas()[0]
        for col in col_categoricas:
            print(f"La distribución de las categorías para la columna {col.upper()}")
            display(self.dataframe[col].value_counts(normalize=True))
            print("........................")

    def imputar_nulos_categoricas(self, lista_moda, lista_nueva_cat):
        """
        Imputa los valores nulos en las columnas categóricas.

        Parámetros:
        - lista_moda: Lista de nombres de columnas donde se imputarán los valores nulos con la moda.
        - lista_nueva_cat: Lista de nombres de columnas donde se imputarán los valores nulos con una nueva categoría "Unknown".

        Retorna:
        - DataFrame: DataFrame con los valores nulos imputados.
        """
        # Imputar valores nulos con moda
        moda_diccionario = {col: self.dataframe[col].mode()[0] for col in lista_moda}
        self.dataframe.fillna(moda_diccionario, inplace=True)

        # Imputar valores nulos con "Unknown"
        self.dataframe[lista_nueva_cat] = self.dataframe[lista_nueva_cat].fillna("Unknown")
    
        return self.dataframe
    
    def identificar_nulos_numericas(self, tamano_grafica=(20, 15)):
        """
        Identifica y visualiza valores nulos en las columnas numéricas mediante gráficos de caja.

        Parámetros:
        - tamano_grafica: Tamaño de las gráficas de caja.
        """
        col_numericas = self.seleccionar_columnas_nulas()[1]

        num_cols = len(col_numericas)
        num_filas = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, col in enumerate(col_numericas):
            sns.boxplot(x=col, data=self.dataframe, ax=axes[indice])
            
        plt.tight_layout();

    def imputar_knn(self, lista_columnas_knn):
        """
        Imputa los valores nulos en las columnas numéricas utilizando el algoritmo KNN.

        Parámetros:
        - lista_columnas_knn: Lista de nombres de columnas numéricas donde se imputarán los valores nulos.

        Retorna:
        - DataFrame: DataFrame con los valores nulos imputados.
        """
        imputer_knn = KNNImputer(n_neighbors=5)
        knn_imputado = imputer_knn.fit_transform(self.dataframe[lista_columnas_knn])

        nuevas_columnas_knn = [col + "_knn" for col in lista_columnas_knn]
        self.dataframe[nuevas_columnas_knn] = knn_imputado

        return self.dataframe
    
    def imputar_imputer(self, lista_columnas_iterative):
        """
        Imputa los valores nulos en las columnas numéricas utilizando el método IterativeImputer.

        Parámetros:
        - lista_columnas_iterative: Lista de nombres de columnas numéricas donde se imputarán los valores nulos.

        Retorna:
        - DataFrame: DataFrame con los valores nulos imputados.
        """
        imputer_iterative = IterativeImputer(max_iter=20, random_state=42)
        iterative_imputado = imputer_iterative.fit_transform(self.dataframe[lista_columnas_iterative])

        nuevas_columnas_iter = [col + "_iterative" for col in lista_columnas_iterative]
        self.dataframe[nuevas_columnas_iter] = iterative_imputado

        return self.dataframe
    
    def comparar_metodos(self):
        """
        Compara los resultados de imputación de los métodos KNN y IterativeImputer.
        """
        columnas_seleccionadas = self.dataframe.columns[self.dataframe.columns.str.contains("_knn|_iterative")].tolist() + self.seleccionar_columnas_nulas()[1].tolist()
        resultados = self.dataframe.describe()[columnas_seleccionadas].reindex(sorted(columnas_seleccionadas), axis=1)
        return resultados

    def columnas_eliminar(self, lista_columnas_eliminar):
        return self.dataframe.drop(lista_columnas_eliminar, axis = 1, inplace = True)