import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import zscore

def exploracion_dataframe(dataframe: pd.DataFrame, columna_control: str):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    #for categoria in dataframe[columna_control].unique():
    #    dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]
    #
    #    print("\n ..................... \n")
    #    print(f"Los principales estadísticos de las columnas categóricas para el {categoria} son: ")
    #    display(dataframe_filtrado.describe(include = "O").T)
        
    #    print("\n ..................... \n")
    #    print(f"Los principales estadísticos de las columnas numéricas para el {categoria} son: ")
    #    display(dataframe_filtrado.describe().T)

def separar_dataframe(dataframe: pd.DataFrame):
    return dataframe.select_dtypes(include = np.number), dataframe.select_dtypes(include = "O")

def plot_numericas(dataframe: pd.DataFrame):
    df_num = separar_dataframe(dataframe)[0]
    cols_numericas = df_num.columns
    num_filas = math.ceil(len(cols_numericas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=(15,10))
    axes = axes.flat
    for indice, columna in enumerate(cols_numericas):
        sns.histplot(x = columna, data=df_num, ax=axes[indice], bins=100)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(cols_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    plt.tight_layout()

def plot_categoricas(dataframe: pd.DataFrame, cols_categoricas: list[str], paleta = "mako", tamanio_grafica = (15, 10)):
    #df_cat = separar_dataframe(dataframe)[1]
    #cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamanio_grafica)
    axes = axes.flat
    for indice, columna in enumerate(cols_categoricas):
        sns.countplot(x = columna, data=dataframe, ax=axes[indice], palette=paleta,
                      order = dataframe[columna].value_counts().index)
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(cols_categoricas) % 2 != 0:
        fig.delaxes(axes[-1])
    else:
        pass
    plt.tight_layout()
    plt.xticks(rotation=90)

def matriz_correlacion(dataframe: pd.DataFrame):
    matriz_corr = dataframe.corr(numeric_only = True)
    plt.figure(figsize=(7,7))
    mascara = np.triu(np.ones_like(matriz_corr, dtype=np.bool_))
    sns.heatmap(matriz_corr,
                annot=True,
                vmin=-1,
                vmax=1,
                mask=mascara)
    plt.tight_layout()

def detectar_outliers(dataframe: pd.DataFrame, color="orange", tamanio_grafica=(15,10)):
    df_num = separar_dataframe(dataframe)[0]
    num_filas = math.ceil(len(df_num.columns)/2)
    fig, axes = plt.subplots(ncols=2, nrows=num_filas, figsize=tamanio_grafica)
    axes = axes.flat
    for indice, columna in enumerate(df_num.columns):

        sns.boxplot(x = columna,
                    data=df_num,
                    ax = axes[indice],
                    color=color,
                    flierprops={"markersize": 5, "markerfacecolor": "red"})
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")
    if len(dataframe.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.tight_layout()

def relacion_vr_categoricas(dataframe: pd.DataFrame, variable_respuesta: str, paleta = "mako", tamanio_grafica = (15,10)):
    df_cat = separar_dataframe(dataframe)[1]
    cols_categoricas = df_cat.columns
    num_filas = math.ceil(len(cols_categoricas)/2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamanio_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_categoricas):
        datos_agrupados = dataframe.groupby(columna)[variable_respuesta].mean().reset_index().sort_values(variable_respuesta, ascending=False)
        sns.barplot(x = columna,
                    y = variable_respuesta,
                    data = datos_agrupados,
                    ax = axes[indice],
                    palette = paleta)
        axes[indice].tick_params(rotation=90)
        axes[indice].set_title(f"Relación entre {columna} y {variable_respuesta}")
        axes[indice].set_xlabel("")
    plt.tight_layout()

def relacion_vr_numericas(dataframe: pd.DataFrame, variable_respuesta: str, paleta = "mako", tamanio_grafica = (15,10)):
    numericas = separar_dataframe(dataframe)[0]
    cols_numericas = numericas.columns
    num_filas = math.ceil(len(cols_numericas)/2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamanio_grafica)
    axes = axes.flat

    for indice, columna in enumerate(cols_numericas):
        if columna == variable_respuesta:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(x = columna,
                        y = variable_respuesta,
                        data = numericas,
                        ax = axes[indice],
                        palette = paleta)
    plt.tight_layout()


def plot_outliers_univariados(dataframe: pd.DataFrame, columnas_numericas, tipo_grafica, bins = 50, whis = 1.5):

    fig, axes = plt.subplots(nrows = math.ceil(len(columnas_numericas) / 2), ncols=2, figsize = (15, 10))

    axes = axes.flat

    for indice, columna in enumerate(columnas_numericas):
        if tipo_grafica == "h":
            sns.histplot(x = columna,
                         data=dataframe,
                         ax = axes[indice],
                         bins = bins)
        elif tipo_grafica == "b":
            sns.boxplot(x = columna,
                        data = dataframe,
                        ax = axes[indice],
                        whis = whis,
                        flierprops = {"markersize": 4, "markerfacecolor": "red"})
        else:
            print("No has elegido una gráfica correcta")

        axes[indice].set_title(f"Distribución columna {columna}")
        axes[indice].set_xlabel("")
        
    if len(columnas_numericas) % 2 != 0:
        fig.delaxes(axes[-1])
            
    plt.tight_layout()


def identificar_outliers_iq(dataframe, columnas_numericas, k = 1.5):
    diccionario_outliers = {}
    for columna in  columnas_numericas:
        q1, q3 = np.nanpercentile(dataframe[columna], (25, 75))
        iqr = q3 - q1

        limite_superior = q3 + (iqr * k)
        limite_inferior = q1 - (iqr * k)

        condicion_sup = dataframe[columna] > limite_superior
        condicion_inf = dataframe[columna] < limite_inferior
    
        df_outliers = dataframe[condicion_inf | condicion_sup]

        print(f"La columna {columna} tiene {df_outliers.shape[0]} outliers")

        if not df_outliers.empty:
            diccionario_outliers[columna] = df_outliers

    return diccionario_outliers


def indentificar_outliers_z(dataframe, columnas_numericas, limite_desviaciones = 3):
    diccionario_outliers = {}
    for columna in columnas_numericas:
        condicion_score = abs(zscore(dataframe[columna])) >= limite_desviaciones
        df_outliers = dataframe[condicion_score]
        print(f"La columna {columna} tiene {df_outliers.shape[0]} outliers")

        if not df_outliers.empty:
            diccionario_outliers[columna] = df_outliers

    return diccionario_outliers