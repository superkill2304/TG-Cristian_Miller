from math import sqrt, ceil
from re import A
from typing import Iterable
from pydataxm.pydataxm import ReadDB
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import col
import scipy as sp
import seaborn as sns
import numpy as np
from scipy.stats import norm
import math
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from traitlets import default
from yarl import SimpleQuery

objectoAPI = ReadDB()


def power_demand(timeoffset="anual", agent=None, iterations=1_000):

    """
    Esta función se encarga de simular la demanda de energía en kWh de los clientes 
    de un comercializador. Permite seleccionar el periodo de análisis y el agente 
    específico del cual se desea obtener la demanda. Retorna una simulación de la 
    demanda horaria basada en una distribución uniforme entre los valores mínimo 
    y máximo observados.
    
    Parámetros:
    - timeoffset: periodo de consulta ("anual", "mensual", o "manual")
    - agent: identificador del agente comercializador
    - iterations: número de simulaciones por hora
    """

    # Fecha de fin del periodo de análisis (hoy)

    end = date.today()

    # Determina la fecha de inicio dependiendo del tipo de periodo elegido

    if timeoffset == "anual":

        start = end - relativedelta(years=1)

    elif timeoffset == "mensual":
        start = end - relativedelta(months=1)

    elif timeoffset == "manual":
        start = str(input("Introduzca la fecha de inicio del período (AAAA-MM-DD): "))
        end = str(input("Introduzca la fecha de fin del período (AAAA-MM-DD): "))
    
    # Solicita los datos de demanda real para el agente desde la API

    df_original = objectoAPI.request_data("DemaReal", "Agente", start, end).set_index(
        "Values_code"
    )

    # Verifica que el código del agente exista en los datos descargados

    if agent in df_original.index:
        pass
    else:
        raise IndexError("No se pudo encontrar el código del agente.")

    df_agent = df_original.loc[agent].copy()

    # Se toman únicamente las columnas correspondientes a las 24 horas del día

    df_agent = df_agent.iloc[:, 1:25]

    # Se obtiene el último registro disponible (demanda actual)

    current = df_agent.tail(1)

    # Se crea un DataFrame con el valor mínimo y máximo de demanda por hora 

    hourly_data = pd.DataFrame({"min": df_agent.min(), "max": df_agent.max()})
    
    # Diccionario para almacenar las simulaciones por hora

    simulated_demand = {}

     # Se itera sobre cada hora (índice) y se generan valores aleatorios uniformes

    for index, data in hourly_data.iterrows():

        simulated_demand[index] = np.random.uniform(
            low=data[0], high=data[1], size=iterations
        )
    # Se convierte el diccionario en un DataFrame donde cada columna es una hora
    simulated_demand = pd.DataFrame(simulated_demand)
    
    

    return simulated_demand, current


def spot_price(timeoffset="anual", iterations=1_000):

    """
    Esta función simula el precio de la electricidad en el mercado spot (precio en bolsa).
    Utiliza una distribución normal basada en las variaciones marginales históricas de precios.

    Parámetros:
    - timeoffset: Periodo de análisis ("anual", "mensual", "manual")
    - iterations: Número de simulaciones que se desea generar

    Retorna:
    - simulated_price: Serie de precios spot simulados
    - current_price: Último precio promedio real disponible
    """
     # Fecha final del período de análisis: hoy 

    end = date.today()

    # Determina la fecha de inicio según el periodo seleccionado

    if timeoffset == "anual":

        start = end - relativedelta(years=1)

    elif timeoffset == "mensual":
        start = end - relativedelta(months=1)

    elif timeoffset == "manual":
        start = str(input("Introduzca la fecha de inicio del período (AAAA-MM-DD): "))
        end = str(input("Introduzca la fecha de fin del período (AAAA-MM-DD): "))

    # Solicita los precios históricos al API de XM para el sistema 

    df_original = objectoAPI.request_data(
        "PrecBolsNaci", "Sistema", start, end
    ).set_index("Date")

    # Se extraen las columnas correspondientes a las 24 horas del día

    df_spot = df_original[df_original.columns[2:26]].copy()
    
    # Se calcula el precio promedio diario (promedio de las 24 horas)

    df_spot["mean_price"] = df_spot.mean(axis=1)

    # Se calcula la variación marginal diaria (variación absoluta de cambio día a día)

    df_spot["marginal"] = (
        df_spot["mean_price"] - df_spot["mean_price"].shift(1)
    ) / df_spot["mean_price"].shift(1)

    # Se llenan valores nulos (por ejemplo, el primer valor de variación) con cero
    
    df_spot.fillna(0, inplace=True)

    # Se simulan nuevas variaciones marginales con una distribución normal

    simulated_marginal_prices = np.random.normal(
        loc=df_spot.marginal.mean(), scale=df_spot.marginal.std(), size=iterations
    )

    # Último precio promedio real disponible

    current_price = df_spot.mean_price.tail(1).iloc[0]

    # Se generan precios simulados multiplicando el precio actual por (1 + variación)
    
    simulated_price = pd.Series(current_price * (simulated_marginal_prices + 1))

    return simulated_price, current_price

def earnings(client_demand,contracted_power,sell_price,market_price,buy_price,C_o,sample_size=1):

    """
    Esta función calcula la utilidad de un comercializador de energía
    según sus dos casos operativos para cada hora del día.

    Parámetros:
    - client_demand: demanda real o simulada de los clientes (valor por hora)
    - contracted_power: potencia contratada (valor por hora) 
    - sell_price: precio de venta a los clientes (valor por hora)
    - market_price: serie o valor fijo con el precio spot del mercado (puede ser iterable o escalar)
    - buy_price: precio de compra al generador (valor por hora)
    - C_o: costos operativos fijos
    - sample_size: número de escenarios simulados

    Retorna:
    - earning_dataframe: DataFrame con las utilidades netas por hora y el total para cada escenario
    """

    # Crea un DataFrame vacío para almacenar las ganancias por hora y por escenario

    earning_dataframe = pd.DataFrame(index=range(sample_size),columns=client_demand.columns)

    # Itera sobre cada hora del día 

    for hour in client_demand.columns:
        Q_d = client_demand[hour].values
        Q_c = contracted_power[hour]
        P_d = sell_price[hour]
        P_c = buy_price[hour]

        # Determina si el precio de mercado es iterable (simulado) o constante (precio actual)
        if isinstance(market_price,Iterable):
            P_b = market_price.values
        else:
            P_b = market_price

        # Calcula el ingreso según el escenario:

        income = np.where(Q_c - Q_d >= 0, Q_d * P_d + (Q_c - Q_d) * P_b, Q_d * P_d)

        # Calcula el costo según el escenario:
        cost = np.where(
            Q_c - Q_d >= 0, Q_c * P_c + C_o, Q_c * P_c + (Q_d - Q_c) * P_b + C_o
        )

        # Calcula la utilidad 

        earning_dataframe[hour] = income - cost 

     # Se agrega una columna con la ganancia total sumando todas las horas
     #   
    earning_dataframe["Total"] = earning_dataframe.sum(axis=1) 

    return earning_dataframe

def main():

    sample_size = int(input("Seleccione el tamaño de muestra:\n" "--> "))

    agent = input("Seleccione el agente de interés:\n" "--> ")

    timeoffset = input(
        "Seleccione el intervalo de tiempo de su interés:\n"
        "  - Digite 'anual'   para un año completo\n"
        "  - Digite 'mensual' para un mes\n"
        "  - Digite 'manual'  para otro periodo personalizado\n"
        "--> "
    )

    option = input(
        "¿El precio de venta a los usuarios esta discriminado por horas??\n"
        "Si: digite y\n"
        "No: digito n\n"
    )

    sell_price = {}

    initializer = None

    for i in range(1, 25):
        hour_label = f"Values_Hour{i:02d}"
        if option == "y":
            value = float(
                input(
                    f"Ingrese el precio de venta de energía para la hora {i:02d}:00 --> $"
                )
            )
            sell_price[hour_label] = value

        elif option == "n":
            if not initializer:
                default = float(
                    input(
                        "Ingrese el precio único de venta de energía para todas las horas --> $"
                    )
                )
                sell_price[hour_label] = default
                initializer = True
            else:
                sell_price[hour_label] = default

        else:
            raise ValueError("Opción inválida")

    

    option = input(
        "¿La cantidad de energia contratada con los generadores esta discriminada por horas?\n"
        "Si: digite y\n"
        "No: digito n\n"
    )

    contracted_power = {}
    initializer = None

    for i in range(1, 25):
        hour_label = f"Values_Hour{i:02d}"
        if option == "y":
            value = float(
                input(
                    f"Ingrese la potencia contratada para la hora {i:02d}:00 (kwh) --> "
                )
            )
            contracted_power[hour_label] = value

        elif option == "n":
            if not initializer:
                default = float(
                    input(
                        "Ingrese la potencia unica contratada para cada hora del dia (kwh) --> "
                    )
                )
                contracted_power[hour_label] = default
                initializer = True
            else:
                contracted_power[hour_label] = default

        else:
            raise ValueError("Opción inválida")

    option = input(
        "¿El precio de la energia comprada a los generadores esta discriminada por horas?\n"
        "Si: digite y\n"
        "No: digito n\n"
    )

    buy_price = {}

    initializer = None

    for i in range(1, 25):
        hour_label = f"Values_Hour{str(i).zfill(2)}"
        if option == "y":
            value = float(
                input(
                    f"Ingrese el precio de compra de la energía para la hora {i:02d}:00 --> $"
                )
            )
            buy_price[hour_label] = value

        elif option == "n":
            if not initializer:
                default = float(
                    input(
                        "Ingrese el precio único de compra de energía para todas las horas --> $"
                    )
                )
                initializer = True
                buy_price[hour_label] = default
            else:
                buy_price[hour_label] = default

        else:
            raise ValueError("Opción inválida")
    operational_cost = float(input("Ingrese el costo operativo:\n" "-->"))

    simulated_demand, today_demand = power_demand(timeoffset, agent, sample_size)
    simulated_price, today_spot = spot_price(timeoffset, sample_size)

    today_earning = earnings(today_demand,contracted_power,sell_price,today_spot,buy_price,operational_cost)
    simulated_earning = earnings(simulated_demand,contracted_power,sell_price,simulated_price,buy_price,operational_cost,sample_size)

    marginal_earning = simulated_earning - today_earning.squeeze()

    print(marginal_earning['Total'].describe())
    return marginal_earning





def plot(item,confidence=95,option='Total'):

    if confidence > 100 and confidence <= 0:
        raise ValueError("Nivel de confianza invalido")
    else:
        quantile = np.quantile(item[option],1 - confidence / 100)
        print(1 - confidence / 100)
    num_of_rows = len(item)
    bins = int(ceil(sqrt(num_of_rows)))

    if option =='Total':

        fig, ax = plt.subplots()
        histogram = plt.hist(item[option],bins=bins,color='blue')

        for rect in histogram[2]:
            if rect.get_x() <= quantile:
                rect.set_facecolor('#FF8000')
        
        
        ax.axvline(x=quantile, linestyle='--', color='black', label='VaR = ${:,.2f}'.format(quantile))
        ax.xaxis.set_major_formatter(lambda x , pos: str(x / 1_000_000))
        plt.title('Histograma de escenarios',fontdict={'fontsize':18,'fontweight':700})
        plt.xlabel("Utilidad marginal (millones $)",fontdict={'fontweight':700})
        plt.ylabel("Frecuencia",fontdict={'fontweight':700})
        plt.legend(loc='upper right')


    sns.despine()
    plt.tight_layout()
    plt.show()
# plot(main())

simulated, x = power_demand(agent='EPSC')
print(simulated.describe())