from math import sqrt, ceil
from typing import Iterable
from pydataxm.pydataxm import ReadDB
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import col, line
import seaborn as sns
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
from traitlets import default
from pathlib import Path
import os

objectoAPI = ReadDB()


def power_demand(start, end, agent=None, iterations=1_000):
    """
    Esta función se encarga de simular la demanda de energía en kWh de los clientes
    de un comercializador. Permite seleccionar el periodo de análisis y el agente
    específico del cual se desea obtener la demanda. Retorna una simulación de la
    demanda horaria basada en una distribución uniforme entre los valores mínimo
    y máximo observados.

    Parámetros:
    - start: inicio de período
    - end: final de período
    - agent: identificador del agente comercializador
    - iterations: número de simulaciones por hora
    """

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


def spot_price(start, end, iterations=1_000):
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


def earnings(
    client_demand,
    contracted_power,
    sell_price,
    market_price,
    buy_price,
    C_o,
    sample_size=1,
):
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

    earning_dataframe = pd.DataFrame(
        index=range(sample_size), columns=client_demand.columns
    )

    # Itera sobre cada hora del día

    for hour in client_demand.columns:
        Q_d = client_demand[hour].values
        Q_c = contracted_power[hour]
        P_d = sell_price[hour]
        P_c = buy_price[hour]

        # Determina si el precio de mercado es iterable (simulado) o constante (precio actual)
        if isinstance(market_price, Iterable):
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
    end = date.today()

    if timeoffset == "anual":

        start = end - relativedelta(years=1)

    elif timeoffset == "mensual":
        start = end - relativedelta(months=1)

    elif timeoffset == "manual":
        start = str(input("Introduzca la fecha de inicio del período (AAAA-MM-DD): "))
        end = str(input("Introduzca la fecha de fin del período (AAAA-MM-DD): "))

    linebreak()
    sell_price = auxiliar("sell_price")
    linebreak()
    contracted_power = auxiliar("contracted_power")
    linebreak()
    buy_price = auxiliar("buy_price")
    linebreak()
    operational_cost = float(input("Ingrese el costo operativo:\n" "--> $"))

    simulated_demand, today_demand = power_demand(start, end, agent, sample_size)
    simulated_price, today_spot = spot_price(start, end, sample_size)

    today_earning = earnings(
        today_demand,
        contracted_power,
        sell_price,
        today_spot,
        buy_price,
        operational_cost,
    )
    simulated_earning = earnings(
        simulated_demand,
        contracted_power,
        sell_price,
        simulated_price,
        buy_price,
        operational_cost,
        sample_size,
    )

    marginal_earning = simulated_earning - today_earning.squeeze()

    linebreak()

    save(marginal_earning)

    linebreak()
    (
        queried_data,
        mean,
        std,
        min,
        max,
        quantile25,
        quantile50,
        quantile75,
        quantileVaR,
        label,
    ) = query(marginal_earning)

    linebreak()
    info(mean, std, min, max, quantile25, quantile50, quantile75, quantileVaR)
    linebreak()

    linebreak()
    plot(queried_data, quantileVaR, label)
    return marginal_earning


def plot(item, quantile, label):
    """Esta función genera un histograma para visualizar la distribución de un conjunto
    de escenarios simulados (por ejemplo, utilidades), y destaca visualmente el valor
    en riesgo (VaR) usando una línea vertical y barras coloreadas hasta ese valor.

    Parámetros:
    - item: Serie con los escenarios simulados (e.g., ganancias).
    - quantile: Valor de corte (VaR) que se quiere marcar en el gráfico.
    - label: Etiqueta para el título del gráfico, que describe el conjunto simulado.
    """
    # Determina el número de filas (escenarios) en el conjunto de datos
    num_of_rows = len(item)

    # Calcula el número de bins del histograma usando la regla de la raíz cuadrada
    bins = int(ceil(sqrt(num_of_rows)))

    # Crea una figura y ejes con Matplotlib
    fig, ax = plt.subplots()

    # Dibuja el histograma con las barras azules
    histogram = plt.hist(item, bins=bins, color="blue")

    # Colorea de naranja las barras del histograma que están por debajo o igual al VaR
    for rect in histogram[2]:
        if rect.get_x() <= quantile:
            rect.set_facecolor("#FF8000")

    # Dibuja una línea vertical punteada que marca el valor del VaR
    ax.axvline(
        x=quantile,
        linestyle="--",
        color="black",
        label="VaR = ${:,.2f}".format(quantile),
    )
    # Cambia el formato del eje X a millones de pesos (dividiendo entre 1 millón)
    ax.xaxis.set_major_formatter(lambda x, pos: str(x / 1_000_000))

    # Establece título y etiquetas con estilo
    plt.title(
        f"Histograma de escenarios {label}",
        fontdict={"fontsize": 18, "fontweight": 700},
    )
    plt.xlabel("Utilidad marginal (millones $)", fontdict={"fontweight": 700})
    plt.ylabel("Frecuencia", fontdict={"fontweight": 700})
    plt.legend(loc="upper right")

    # Quita los bordes superiores y derechos (estética)
    sns.despine()

    # Ajusta el diseño para evitar solapamientos
    plt.tight_layout()
    # Muestra el gráfico
    plt.show()


def auxiliar(variable: str):
    """Esta es una función auxiliar para crear 3 tipos de variables:
    - Precio de venta (sell_price)
    - Potencia contratada (contracted_power)
    - Precio de compra (buy_price)
    Retorna un diccionario con los valores por hora (de 1 a 24).
    """

    # Asigna los textos que se mostrarán al usuario según el tipo de variable

    if variable == "sell_price":
        label1 = (
            "¿El precio de venta de energía a los usuarios está discriminado por horas?"
        )
        label2 = "el precio de venta de energía a los usuarios"
        label3 = "Ingrese el precio único de venta de energía para todas las horas (en $/kwh)\n--> $"

    elif variable == "contracted_power":
        label1 = "¿La cantidad de energía contratada con los generadores está discriminada por horas?"
        label2 = "la potencia contratada con los generadores"
        label3 = "Ingrese la potencia única contratada  para cada hora del día (en kWh):\n--> "

    elif variable == "buy_price":
        label1 = "¿El precio de la energía contratada con los generadores está discriminado por horas?"
        label2 = "Ingrese el precio de compra de la energía"
        label3 = "Ingrese el precio único de compra de energía para todas las horas (en $/kwh)\n--> $"
    else:
        # Si se pasa una variable inválida, se lanza un error
        raise ValueError(
            "Allowed values are: ['sell_price', 'contracted_power', 'buy_price']"
        )

    # Diccionario que almacenará los valores por hora

    data = {}

    # Flag para saber si ya se pidió el valor único en caso de opción "n"

    initializer = None

    # Pregunta al usuario si desea discriminar los valores por hora

    option = input(f"{label1}\n\n" "Si: digite y\n" "No: digito n\n" "--> ")

    # Recorre las 24 horas del día

    for i in range(1, 25):
        hour_label = f"Values_Hour{i:02d}"
        if option == "y":
            # Si el usuario quiere valores por hora, se pide uno por cada hora
            value = float(input(f"Ingrese {label2} para la hora {i:02d}:00 \n--> $"))
            data[hour_label] = value

        elif option == "n":
            if not initializer:
                default = float(input(f"{label3}"))
                data[hour_label] = default
                initializer = True
            else:
                # Se reutiliza el mismo valor para el resto de horas
                data[hour_label] = default

        else:
            # Si el usuario digita algo diferente a "y" o "n", lanza un error
            raise ValueError("Opción inválida")

    # Retorna el diccionario con los valores por hora
    return data


def linebreak():
    print(f"{110*'_'}\n")


def query(data):
    """
    Esta función permite al usuario consultar estadísticas sobre la utilidad marginal.
    Puede seleccionar entre la utilidad marginal total diaria o la de una hora específica.
    Retorna los datos consultados junto con medidas estadísticas y un texto descriptivo de la consulta.
    """
    # Solicita al usuario la opción de consulta: total o por hora

    option = input(
        "Seleccione una opción:\n"
        "- Para consultar la utilidad marginal total, digite: t\n"
        "- Para consultar la utilidad marginal en una hora específica, digite el número de la hora (formato 1 a 24).\n"
        "  Ejemplo: Para consultar la utilidad marginal a las 12:00, digite: 12\n"
        "--> "
    )

    # Si el usuario selecciona 't', consulta la columna 'Total'

    if option == "t":
        queried_data = data["Total"]
        label = "Diarios"
    else:
        try:

            # Intenta convertir la opción a entero y verifica si está en el rango 1–24

            if int(option) in range(1, 25):
                hour = int(option)
                queried_data = data[f"Values_Hour{hour:02d}"]
                label = f"{hour:02d}:00"

            else:
                raise ValueError  # Lanza error si el número no está en el rango
        except ValueError:
            raise ValueError(
                "El valor digitado no es válido."
            )  # Error si no es un número o fuera de rango

    # Solicita el nivel de confianza para calcular el Valor en Riesgo (VaR)

    confidence = float(
        input(
            "Ingrese el nivel de confianza que desea utilizar para el cálculo del VaR (entre 0 y 100).\n"
            "Ejemplo: para un 95% de confianza, escriba 95.\n"
            "--> "
        )
    )

    # Ajusta el nivel de confianza para usarlo en el cálculo del cuantil

    if confidence > 100 or confidence <= 0:
        raise ValueError("El nivel de confianza debe ser un número entre 0 y 100.")

    adjusted_confidence = 1 - confidence / 100

    # Retorna los datos consultados y todas las métricas calculadas junto con la etiqueta de la consulta

    mean = queried_data.mean()
    std = queried_data.std()
    min = queried_data.min()
    max = queried_data.max()
    quantile25 = np.quantile(queried_data, 0.25)
    quantile50 = np.quantile(queried_data, 0.5)
    quantile75 = np.quantile(queried_data, 0.75)
    quantileVaR = np.quantile(queried_data, adjusted_confidence)

    return (
        queried_data,
        mean,
        std,
        min,
        max,
        quantile25,
        quantile50,
        quantile75,
        quantileVaR,
        label,
    )


def info(mean, std, min, max, quantile25, quantile50, quantile75, quantileVaR):
    """
    Imprime en consola un resumen estadístico con formato financiero ($) usando 3 decimales.
    Los valores presentados incluyen:
    - Promedio
    - Desviación estándar
    - Mínimo y máximo
    - Percentiles 25%, 50% (mediana), 75%
    - Valor en Riesgo (VaR) calculado para un nivel de confianza especificado
    """

    print(
        "           Promedio: ${:,.3f}\n"
        "Desviación estándar: ${:,.3f}\n"
        "             Mínimo: ${:,.3f}\n"
        "             Máximo: ${:,.3f}\n"
        "      Percentil 25%: ${:,.3f}\n"
        "      Percentil 50%: ${:,.3f}\n"
        "      Percentil 75%: ${:,.3f}\n"
        "                VaR: ${:,.3f}".format(
            mean, std, min, max, quantile25, quantile50, quantile75, quantileVaR
        )
    )


def save(item):
    """
    Solicita al usuario si desea guardar los escenarios simulados (DataFrame 'item') en un archivo .csv.
    Si elige 'y', pide un nombre de archivo y lo guarda en la carpeta 'escenarios'.
    Si el archivo ya existe, solicita un nuevo nombre.
    Si elige 'n', no hace nada.
    Si se ingresa una opción no válida, lanza un error.
    """

    option = input(
        "¿Desea guardar los escenarios simulados?\n\n"
        "Digite 'y' para Sí\n"
        "Digite 'n' para No\n\n"
        "--> "
    )

    if option == "y":
        # Determina el directorio base del archivo actual (es decir, la carpeta donde se encuentra modelo.py)
        base_dir = Path(__file__).resolve().parent

        escenarios_folder = base_dir / "escenarios"

        if not os.path.exists(escenarios_folder):

            os.makedirs(escenarios_folder)

        while True:

            # Pide al usuario un nombre de archivo y añade la extensión .csv
            file_name = input("Nombre para guardar el archivo: ") + ".csv"
            full_path = escenarios_folder / file_name

            if not os.path.exists(full_path):
                item.to_csv(full_path)
                break
            else:
                print(
                    f"⚠️ El archivo '{file_name}' ya existe en la carpeta: {escenarios_folder}\n"
                    "Por favor, ingrese un nuevo nombre o elimine el archivo existente para continuar. \n"
                )
        print(f"✔️ Archivo '{file_name}' creado con éxito.")

    elif option == "n":
        return None

    else:
        raise ValueError(f"Invalid option {option}")


if __name__ == "__main__":

    print(
    "\nSCRIPT DESARROLLADO POR CRISTIAN DAVID MILLER GONZÁLEZ\n"
    "PARA EL TRABAJO DE GRADO SOBRE RIESGOS FINANCIEROS EN MERCADOS ELÉCTRICOS\n"
    "email: cristian.miller@correounivalle.edu.co"
)

    main()
