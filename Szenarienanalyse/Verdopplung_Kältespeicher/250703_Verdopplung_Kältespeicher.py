# Szenario Verdopplung Kältespeicher


# 1. Grundlagen


# 1.1 Import der Bibliotheken


import fine as fn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import csv
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import datetime as dt
from IPython.display import display


cwd = Path.cwd()

np.random.seed(
    42
)  # Sets a "seed" to produce the same random input data in each model run

import locale

# Versuche die Sprachumgebung auf Deutsch zu setzen
try:
    locale.setlocale(locale.LC_TIME, 'de_DE.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_TIME, 'de_DE')
    except locale.Error:
        print("Locale 'de_DE' not found. Proceeding with default locale.")
# -------------------------------------------------------------
# 1.2 Colorcoding


color_strom = (0 / 255, 83 / 255, 116 / 255)  # RGB (0, 83, 116)
color_raumwaerme = (176 / 255, 0 / 255, 70 / 255)  # RGB (176, 0, 70)
color_kälte = (100/255, 50/255, 200/255)  # RGB (176, 0, 70)

gelb_pv = (255 / 255, 205 / 255, 0 / 255)
orange_wp = (250 / 255, 110 / 255, 0 / 255)
rot_raumwaerme = color_raumwaerme
hellblau_lib = (124 / 255, 205 / 255, 230 / 255)
blau_stromkauf = (0 / 255, 128 / 255, 180 / 255)
dunkelblau_strom = color_strom
grün_tes = (0 / 255, 113 / 255, 86 / 255)
lila_gas = (118 / 255, 0 / 255, 118 / 255)
weinrot_bhkw = (118 / 255, 0 / 255, 84 / 255)
hellgruen_AKM = ((50/255, 200/255, 125/255))
pink_KKM =((200/255, 50/255, 125/255))

color_raumwaerme = (176 / 255, 0 / 255, 70 / 255)  # RGB (176, 0, 70)
color_raumwaerme_light = (243 / 255, 217 / 255, 227 / 255)
color_strom = (0 / 255, 83 / 255, 116 / 255)
color_strom_light = (217 / 255, 229 / 255, 234 / 255)

farben_komponenten = {
    "Netzanschlussleistung": color_strom,
    "PV": gelb_pv,
    "BHKW": weinrot_bhkw,
    "Wärmepumpe": orange_wp,
    "AKM": hellgruen_AKM,
    "Absorptionskältemaschine": hellgruen_AKM,
    "Kompressionskältemaschine": pink_KKM,
    "KKM": pink_KKM,
    "Heizungswasserspeicher": grün_tes,
    "Kältespeicher": color_kälte,
    "LIB": hellblau_lib,
    "Lithium-Ionen-Batterie": hellblau_lib,
    "Erdgaskauf": lila_gas,
    "Stromkauf": blau_stromkauf
}

#custom_cmap_rot = LinearSegmentedColormap.from_list("weiß_zu_raumwaerme",[color_raumwaerme_light, color_raumwaerme])
custom_cmap_rot = LinearSegmentedColormap.from_list("weiß_zu_raumwaerme", [color_strom, gelb_pv])

# custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme", [(124/255, 205/255, 239/255), (190/255, 30/255, 70/255)])
custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme",
                                                        [color_strom, color_raumwaerme_light, color_raumwaerme])
# -------------------------------------------------------------
# 1.3 Einlesen der pickle-Dateien aus dem "Inputs"-Skript


stromverbr = pd.read_pickle("stromverbrauch.pkl")
heizungswasserverb = pd.read_pickle("heizungswasser.pkl")
strompreis_cost = pd.read_pickle("strompreis_cost.pkl")
strompreis_revenue = pd.read_pickle("strompreis_revenue.pkl")           #aktuell keine negativen Strompreise: druchschn. 33,75 €/kWh
strompreis_cost_sinkt = pd.read_pickle("strompreis_cost_sinkt.pkl")
strompreis_revenue_sinkt = pd.read_pickle("strompreis_revenue_sinkt.pkl")
strompreis_cost_steigt = pd.read_pickle("strompreis_cost_steigt.pkl")
strompreis_revenue_steigt = pd.read_pickle("strompreis_revenue_steigt.pkl")
strompreis_cost_schwankt = pd.read_pickle("strompreis_cost_schwankt.pkl")
strompreis_revenue_schwankt = pd.read_pickle("strompreis_revenue_schwankt.pkl")
stromemissionen = pd.read_pickle("stromemissionen.pkl")
gaspreis = pd.read_pickle("gaspreis.pkl")
gaspreis_sensi = pd.read_pickle("gaspreisSensi.pkl")
pv = pd.read_pickle("pv.pkl")
cop = pd.read_pickle("cop.pkl")

kältebed=pd.read_pickle("kältebedarf.pkl")
kältebed = kältebed.clip(lower=0)

df_strom_pr_em_2024 = pd.read_pickle("StromPreisEmissionen.pkl") # €/MWh
df_strom_pr_em_2024_sensi = pd.read_pickle("StromPreisEmissionenSensi.pkl")

#stromemissionen.mean()
#stromemissionen=0.143*stromemissionen      # für Sensitivitätsbetrachtung
#stromemissionen.mean()
# -------------------------------------------------------------
# 2. Modellierung in FINE

# 2.1 Initialisierung des "energy system model"


locations = {"FabrikAachen"}
commodityUnitDict = {
    "strom": r"kW_el", # sinnvolle Maßeinheiten pro Stunde
    "netzstrom": r"kW_el",
    "erdgas": r"kW_CH4,LHV",
    "CO2": r"gCO2/h",
    "heizungswasser": r"kW_th",
    "kälte": r"kW_th_c"
}
commodities = {"strom", "netzstrom", "erdgas", "CO2", "heizungswasser","kälte"}
numberOfTimeSteps, hoursPerTimeStep = 8784, 1 # Index geht von 0 bis 8759

costUnit, lengthUnit = "Euro", "m" # 1e3  wären 1000
startYear = 2024



esM = fn.EnergySystemModel(
    locations=locations,
    commodities=commodities,
    numberOfTimeSteps=numberOfTimeSteps,
    commodityUnitsDict=commodityUnitDict,
    hoursPerTimeStep=hoursPerTimeStep,
    costUnit=costUnit,
    lengthUnit=lengthUnit,
    verboseLogLevel=0,
    startYear=startYear,
)

# -------------------------------------------------------------

# 2.2 Anlegen der FINE-Klassen

esM.componentNames

esM.add(
    fn.Source(
        esM=esM,
        name="PV",
        commodity="strom",
        hasCapacityVariable=True,
        # capacityVariableDomain wird als continuous angenommen
        operationRateFix=pv,
        capacityFix=2000, # kW
        #capacityMax=4000,
        investPerCapacity=833, # €/kW
        opexPerCapacity=833 * 0.01, # €/kW
        interestRate=0.05,
        economicLifetime=20, # a
        # yearlyFullLoadHoursMin bzw. Max ist bei gegebener operationRateMax und capacity nicht wichtig
        # commodityRevenue=0.1047 # €/kWh
        commodityCost=0.14  # Contracting der PV-Anlage: 14 ct/kWh
    )
)


esM.add(
    fn.Source(
        esM=esM,
        name="Stromkauf von Spotmarkt",
        commodity="netzstrom",
        hasCapacityVariable=True,
        commodityCostTimeSeries=strompreis_cost, # €/kWh
        commodityRevenueTimeSeries=strompreis_revenue, # €/kWh
        economicLifetime = 1,
        capacityMax = 5000,  # max. Netzanschlussleistung 2 MW
        investPerCapacity = 51.35  # €/kW
        #investPerCapacity = 214.59  # €/kW
    )
)



esM.add(
    fn.Conversion(
        esM=esM,
        name="Netzanschluss",
        physicalUnit=r"kW_el",
        commodityConversionFactors={
            "netzstrom": -1,
            "strom": 1,
            "CO2": stromemissionen, # g/kWh
        },
        hasCapacityVariable=False,
    )
)

co2_faktor_gas = 200.8

esM.add(
    fn.Conversion(
        esM=esM,
        name="BHKW",
        physicalUnit=r"kW_el", # d. h., so skalieren, dass commodityConversionFactors für "strom": 1
        commodityConversionFactors={
            "strom": 1, # nach eigenen Berechnungen, mit Ausreißerbereinigung
            "heizungswasser": 1, # inkl. heisswasser # nach eigenen Berechnungen, mit Ausreißerbereinigung
            "erdgas": -1 / 0.3675,
            "CO2": co2_faktor_gas / 0.3675, # g/kWh_Primärenergie, nach Quasching, wie Jin et. al. // FfE: 50 bis zu 100 g/kWh_el...dort sehr verschiedene Allokationsmethoden // 2017er Paper: 50-100, 200-280
        },
        hasCapacityVariable=True,
        capacityMax=1.99*1000, # Elektrische Nennleistung von 1,99 MW_el, durchschnittliche Leistung von 1,0 (bis April 1,66) MW_el
        interestRate=0.05,
        economicLifetime=20-10, # remainingEconomicLifetime 10a # default 10
        operationRateMin=0.5,
        investPerCapacity=0, # wird in den nächsten Jahren abgeschrieben, 800€/kW_el
        opexPerCapacity=800*0.02, # 16€/kW_el
        opexPerOperation=0.008, # €/kWh_el
        # yearlyFullLoadHoursMin?
    )
)

esM.add(
    fn.Conversion(
        esM=esM,
        name="Absorptionskältemaschine",
        physicalUnit=r"kW_th_c", # commodityConversionFactors so skalieren, dass kälte: 1
        commodityConversionFactors={
            "kälte": 1,
            "heizungswasser": -1/0.7
            },
        hasCapacityVariable=True,
        #capacityMax=420,
        capacityMax=420,           # thermische Nennleistung
        interestRate=0.05,
        economicLifetime=20 - 10,  # remainingEconomicLifetime 10a # default 10
        investPerCapacity=0,
        opexPerCapacity=300 * 0.01,  # €/kW_th
        opexPerOperation=0.015,  # €/kW_th
        hasIsBuiltBinaryVariable=False,     # Anlage besteht bereits
        bigM=420  # Maximalbedarf Raumwärme pro Stunde sind 2858 kWh
    )
)

esM.add(
    fn.Conversion(
        esM=esM,
        name="Kompressionskältemaschine",
        physicalUnit=r"kW_th_c", # commodityConversionFactors so skalieren, dass kälte: 1
        commodityConversionFactors={
            "kälte": 1,
            "strom": -1/2.5
            },
        hasCapacityVariable=True,
        capacityMax=300,
        #capacityMax=600,           # thermische Nennleistung
        interestRate=0.05,
        investPerCapacity=0,  # 100 €/kWh Frauenhofer, bei halber installierter Leistung 50 €/kWh
        #investPerCapacity=1100,      # analog zu WP
        economicLifetime=20 - 10,  # remainingEconomicLifetime 10a # default 10
        opexPerCapacity=100 * 0.01,  # €/kW_el
        opexPerOperation=0.015,  # €/kWh_el
        #opexPerOperation=0.25,  # €/kWh_el
        hasIsBuiltBinaryVariable=False,     # Anlage besteht bereits
        bigM=300  # Maximalbedarf Raumwärme pro Stunde sind 2858 kWh

    )
)

esM.add(
    fn.Conversion(
        esM=esM,
        name="Wärmepumpe",
        physicalUnit=r"kW_th", # commodityConversionFactors so skalieren, dass heizungswasser: 1
        commodityConversionFactors={"strom": -1/cop, "heizungswasser": 1},
        hasCapacityVariable=True,
        investPerCapacity=1100, # €/kW_th # je mehr kW umso billiger
        opexPerCapacity=0.01*1100, # €/kW_th
        interestRate=0.05,
        economicLifetime=17,
        hasIsBuiltBinaryVariable=True,
        opexPerOperation=0.015,
        bigM=3000 # Maximalbedarf Raumwärme pro Stunde sind 2858 kWh
    )
)

esM.add(
    fn.Storage(
        esM=esM,
        name="Heizungswasserspeicher",
        commodity="heizungswasser",
        chargeRate=0.25,
        dischargeRate=0.25,
        chargeEfficiency=0.95,
        dischargeEfficiency=0.95,
        selfDischarge=1 - (1 - 0.1) ** (1 / (8760/2)), # 0.0026
        hasIsBuiltBinaryVariable=True,
        hasCapacityVariable=True,
        interestRate=0.05,
        economicLifetime=30,
        #cyclicLifetime=3000, ?
        investPerCapacity=10, # €/kWh
        opexPerCapacity=0.015*10, # €/kWh
        #bigM=75000, # 19.000 ist maximaler Heizungswasserbedarf in 24h
        bigM=375+2315*3,
        #capacityMax=375
        capacityMax=375+2315*3  # 300 m3 + 16 m3 (Pufferspeicher)
    )
)

esM.add(
    fn.Storage(
        esM=esM,
        name="Kältespeicher",
        commodity="kälte",
        chargeRate=0.25,
        dischargeRate=0.25,
        chargeEfficiency=0.95,
        dischargeEfficiency=0.95,
        selfDischarge=0.0026,
        hasCapacityVariable=True,
        interestRate=0.05,
        economicLifetime=30,
        investPerCapacity=10, # €/kWh
        opexPerCapacity=10*0.015, # €/kWh
        hasIsBuiltBinaryVariable=True,
        bigM=(700*3)*2,
        #capacityMax=700
        capacityMax = (700*3)*2 # 300 m3
    )
)

esM.add(
    fn.Storage(
        esM=esM,
        name="Lithium-Ionen-Batterie",
        commodity="strom",
        cyclicLifetime=100000, # 5000-10000, PRAMAC hat 7300
        economicLifetime=12,
        chargeEfficiency=0.96, # Gesamt 0.87-0.95, PRAMAC hat 0.93
        dischargeEfficiency=0.96,
        chargeRate=0.33, # je kleiner, umso höher (0.3), je größer, umso niedriger (0.17), PRAMAC hat auch große mit großer Ent/ladeleistung 0,8
        dischargeRate=0.33,
        selfDischarge=1 - (1 - 0.05) ** (1 / (30 * 24)),
        hasIsBuiltBinaryVariable=True,
        hasCapacityVariable=True,
        interestRate=0.05,
        investPerCapacity=411.08, # €/kWh
        opexPerCapacity=10.57,
        bigM=2000,
        #bigM=10000, # ca. 3.000 ist max. Strombedarf in einer Stunde und LIB sollte 4 Stunden aushalten
    )
)

esM.add(
    fn.Source(
        esM=esM,
        name="Erdgaskauf",
        commodity="erdgas",
        hasCapacityVariable=False,
        commodityCostTimeSeries=gaspreis # €/kWh
        #commodityCost=0.0336
    )
)

esM.add(
    fn.Sink(
        esM=esM,
        name="Strombedarf",
        commodity="strom",
        hasCapacityVariable=False,
        operationRateFix=stromverbr, # kWh
    )
)

esM.add(
    fn.Sink(
        esM=esM,
        name="Heizungswasserbedarf",
        commodity="heizungswasser",
        hasCapacityVariable=False,
        operationRateFix=heizungswasserverb, # kWh
    )
)

esM.add(
    fn.Sink(
        esM=esM,
        name="Kältebedarf",
        commodity="kälte",
        hasCapacityVariable=False,
        operationRateFix=kältebed, # kWh
    )
)

esM.add(
    fn.Sink(
        esM=esM,
        name="CO2 in Atmosphäre",
        commodity="CO2",
        hasCapacityVariable=False,
        commodityLimitID="CO2 limit",
        yearlyLimit= 7500 * 1000 * 1000, # erste Zahl in t denken und dann umrechnen in g
        #yearlyLimit=5004938271.6049385,
        commodityCost=45/(1000*1000), # 30 €/t CO2-Preis
    )
)

# -------------------------------------------------------------
# 3. Zwischenberechnungen

capacity_kw = 1990
efficiency = 0.4225/0.4225
hours_in_year = 8784

index = range(0, hours_in_year)
generation_series = pd.Series([capacity_kw * efficiency] * len(index), index=index)
stromüberschuss = generation_series + pv*2000 - stromverbr

stromüberschuss = np.clip(stromüberschuss, 0, None)
esM.add(
    fn.Sink(
        esM=esM,
        name="Überschussstrom in Netz",
        commodity="strom",
        hasCapacityVariable=False,
        operationRateMax=stromüberschuss,
        commodityCostTimeSeries=strompreis_revenue,
        #operationRateMax= pd.Series([2300] * len(index), index=index),
    )
)

print(esM.componentNames)

# -------------------------------------------------------------

# 4. Optimierung des Energiesystems

# 4.1 Ausführung der Optimierung

timeSeriesAggregation=False
solver="gurobi"

esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)

# -------------------------------------------------------------

# 4.2 Ausgabe der TAC und THGE

def extract_tac_and_co2(esM):
    """
    Extract TAC and CO2 emissions from the esM results.

    Parameters:
    esM (EnergySystemModel): The energy system model instance after optimization.

    Returns:
    tuple: Total Annual Cost (TAC) and CO2 emissions.
    """
    # Get results for TAC and CO2 emissions

    srcSnkSummary_2 = esM.getOptimizationSummary("SourceSinkModel", ip=2024, outputLevel=2)
    convSummary_2 = esM.getOptimizationSummary("ConversionModel", ip=2024, outputLevel=2)
    storSummary_2 = esM.getOptimizationSummary("StorageModel", ip=2024, outputLevel=2)

    # Initialize TAC values
    tac_srcsnk = 0
    tac_conv = 0
    tac_stor = 0

    # Extract TAC
    if 'TAC' in srcSnkSummary_2.index.get_level_values('Property'):
        tac_srcsnk = srcSnkSummary_2.xs('TAC', level='Property').sum().sum()
    if 'TAC' in convSummary_2.index.get_level_values('Property'):
        tac_conv = convSummary_2.xs('TAC', level='Property').sum().sum()
    if 'TAC' in storSummary_2.index.get_level_values('Property'):
        tac_stor = storSummary_2.xs('TAC', level='Property').sum().sum()
    tac = tac_srcsnk + tac_stor + tac_conv

    # Extract CO2 emissions from the "CO2 to environment" sink
    co2_srcsnk = srcSnkSummary_2.xs(('operation', '[gCO2/h*h/a]'), level=('Property', 'Unit')).loc[
        'CO2 in Atmosphäre'].sum()
    co2_emissions = co2_srcsnk

    return tac, co2_emissions

# -------------------------------------------------------------

# 4.3 Definition multikritielle Zielfunktion

#Erläuterungen

# Grobe Suche: Die Funktion erstellt zunächst eine Liste von yearlyLimit Werten und führt das Modell für jeden dieser Werte aus.
# Bestimmung der besten Paare: Nach der Ausführung aller Tests sucht die Funktion das Paar aufeinanderfolgender yearlyLimit Werte mit den niedrigsten kombinierten Zielfunktionswerten.

# Feine Suche: Nachdem die besten aufeinanderfolgenden Limits identifiziert wurden,
# wird eine detailliertere Suche in diesem engeren Bereich durchgeführt, um das optimale yearlyLimit zu bestimmen.


def compute_objective(esM, limit, weight_co2, weight_tac, max_tac, max_co2, timeSeriesAggregation, solver,
                      precomputed_values):
    if limit not in precomputed_values:
        precomputed_values[limit] = {}
        esM.updateComponent("CO2 in Atmosphäre", {"yearlyLimit": limit})
        if timeSeriesAggregation:
            numberOfTypicalPeriods = 366
            esM.aggregateTemporally(numberOfTypicalPeriods=numberOfTypicalPeriods)
        esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)
        srcSinkSummary_interim = esM.getOptimizationSummary("SourceSinkModel", ip=2024, outputLevel=1)
        convSummary_interim = esM.getOptimizationSummary("ConversionModel", ip=2024, outputLevel=1)
        storSummary_interim = esM.getOptimizationSummary("StorageModel", ip=2024, outputLevel=1)
        timeseries_srcsnk_interim = esM.componentModelingDict[
            esM.componentNames["Stromkauf von Spotmarkt"]].getOptimalValues("operationVariablesOptimum", ip=2024)
        timeseries_srcsnk_interim = timeseries_srcsnk_interim["values"]
        timeseries_stor_discharge_interim = esM.componentModelingDict[
            esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("dischargeOperationVariablesOptimum",
                                                                           ip=2024)
        timeseries_stor_discharge_interim = timeseries_stor_discharge_interim["values"]
        timeseries_stor_charge_interim = esM.componentModelingDict[
            esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("chargeOperationVariablesOptimum", ip=2024)
        timeseries_stor_charge_interim = timeseries_stor_charge_interim["values"]
        timeseries_stor_soc_interim = esM.componentModelingDict[
            esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("stateOfChargeOperationVariablesOptimum",
                                                                           ip=2024)
        timeseries_stor_soc_interim = timeseries_stor_soc_interim["values"]
        timeseries_conv_interim = esM.componentModelingDict[esM.componentNames["BHKW"]].getOptimalValues(
            "operationVariablesOptimum", ip=2024)
        timeseries_conv_interim = timeseries_conv_interim["values"]
        tac, co2_emissions = extract_tac_and_co2(esM)
        objective_value = weight_tac * (tac / max_tac) + weight_co2 * (co2_emissions / max_co2)
        precomputed_values[limit][weight_co2] = (
        objective_value, tac, co2_emissions, srcSinkSummary_interim, convSummary_interim, storSummary_interim,
        timeseries_srcsnk_interim, timeseries_conv_interim, timeseries_stor_charge_interim,
        timeseries_stor_discharge_interim, timeseries_stor_soc_interim)
    elif weight_co2 not in precomputed_values[limit]:
        tac, co2_emissions, srcSinkSummary_interim, convSummary_interim, storSummary_interim, timeseries_srcsnk_interim, timeseries_conv_interim, timeseries_stor_charge_interim, timeseries_stor_discharge_interim, timeseries_stor_soc_interim = \
        precomputed_values[limit][next(iter(precomputed_values[limit]))][1:]
        # next() fetches the next item from the iterator. Since it's the first call to next(), it gets the first key in the dictionary of precomputed_values[limit]. This key represents a weight_co2 value that was previously computed and stored. This slices the tuple to extract the second and third elements, which are tac and co2_emissions.
        objective_value = weight_tac * (tac / max_tac) + weight_co2 * (co2_emissions / max_co2)
        precomputed_values[limit][weight_co2] = (
        objective_value, tac, co2_emissions, srcSinkSummary_interim, convSummary_interim, storSummary_interim,
        timeseries_srcsnk_interim, timeseries_conv_interim, timeseries_stor_charge_interim,
        timeseries_stor_discharge_interim, timeseries_stor_soc_interim)
    return precomputed_values[limit][weight_co2]


def optimization_heuristic(esM, min_co2, max_co2, num_trials, max_tac, weight_co2, epsilon, timeSeriesAggregation,
                           solver, precomputed_values):
    weight_tac = 1 - weight_co2

    # Schritt 1: Grobe Suche
    limits = np.linspace(min_co2, max_co2, num_trials + 2)
    results = []

    for limit in limits:
        objective_value, tac, co2_emissions, srcSinkSummary_interim, convSummary_interim, storSummary_interim, timeseries_srcsnk_interim, timeseries_conv_interim, timeseries_stor_charge_interim, timeseries_stor_discharge_interim, timeseries_stor_soc_interim = compute_objective(
            esM, limit, weight_co2, weight_tac, max_tac, max_co2, timeSeriesAggregation, solver, precomputed_values)
        results.append((objective_value, limit, tac, co2_emissions))

    print(results)

    # Finde die zwei nebeneinander liegenden besten Ergebnisse
    best_pair_index = None
    best_pair_value = float('inf')

    for i in range(len(results) - 1):
        combined_value = results[i][0] + results[i + 1][0]
        if combined_value < best_pair_value:
            best_pair_value = combined_value
            best_pair_index = i

    best_two_limits = [results[best_pair_index][1], results[best_pair_index + 1][1]]

    # Schritt 2: Feine Suche zwischen den zwei besten Limits
    fine_limits = np.linspace(min(best_two_limits), max(best_two_limits),
                              num_trials + 2)  # Feinere Suche in einem engeren Bereich
    fine_results = []

    for limit in fine_limits:
        objective_value, tac, co2_emissions, srcSinkSummary_interim, convSummary_interim, storSummary_interim, timeseries_srcsnk_interim, timeseries_conv_interim, timeseries_stor_charge_interim, timeseries_stor_discharge_interim, timeseries_stor_soc_interim = compute_objective(
            esM, limit, weight_co2, weight_tac, max_tac, max_co2, timeSeriesAggregation, solver, precomputed_values)
        fine_results.append((objective_value, limit, tac, co2_emissions, srcSinkSummary_interim, convSummary_interim,
                             storSummary_interim, timeseries_srcsnk_interim, timeseries_conv_interim,
                             timeseries_stor_charge_interim, timeseries_stor_discharge_interim,
                             timeseries_stor_soc_interim))

        if len(fine_results) > 1 and abs(fine_results[-1][0] - fine_results[-2][0]) < epsilon:
            print("Epsilon-Kriterium erreicht. Beendigung der Suche.")
            break

    fine_results.sort()
    print(fine_results)
    best_result = fine_results[0]

    print(
        f"Optimales yearlyLimit: {best_result[1]} mit einem Zielfunktionswert von {best_result[0]} bei einem CO2-Gewicht von {weight_co2}")

    return best_result

# -------------------------------------------------------------

# 4.4 Festlegung von TAC-/THGE-Obergrenzen

#min_co2 = 1500 * 1000 * 1000  # g Co2 pro Jahr, erste Zahl in Tonnen denken
min_co2 = 1  # g Co2 pro Jahr, erste Zahl in Tonnen denken

max_co2 = 7500 * 1000 * 1000  # g Co2 pro Jahr, erste Zahl in Tonnen denken
#max_tac = 2842 * 1000  # erste Zahl in Tausend € pro Jahr denken
#max_tac = 6500 * 1000  # erste Zahl in Tausend € pro Jahr denken
max_tac = 6000 * 1000  # erste Zahl in Tausend € pro Jahr denken
weight_values = np.linspace(0, 0.4, 11)  # Gewichtungen von 0 bis 1 in Schritten von 0.1

# -------------------------------------------------------------
# 4.5 Abspeichern der Simulationsergebnisse in Excel-Sheets


pareto_results = []
precomputed_values = {}

for weight_co2 in weight_values:
    best_heuristic_result = optimization_heuristic(esM, min_co2, max_co2, 12, max_tac, weight_co2, 0.001, timeSeriesAggregation, solver, precomputed_values)
    pareto_results.append((*best_heuristic_result[:4], weight_co2))
    srcSinkSummary_interim, convSummary_interim, storSummary_interim, timeseries_srcsnk_interim, timeseries_conv_interim, timeseries_stor_charge_interim, timeseries_stor_discharge_interim, timeseries_stor_soc_interim = best_heuristic_result[4:]
    srcSinkSummary_interim.to_csv(f"srcSinkSummary_weight_co2_{weight_co2}.csv")
    convSummary_interim.to_csv(f"convSummary_weight_co2_{weight_co2}.csv")
    storSummary_interim.to_csv(f"storSummary_weight_co2_{weight_co2}.csv")
    timeseries_srcsnk_interim.to_csv(f"srcSinkTS_weight_co2_{weight_co2}.csv")
    timeseries_conv_interim.to_csv(f"convTS_weight_co2_{weight_co2}.csv")
    timeseries_stor_charge_interim.to_csv(f"storChargeTS_weight_co2_{weight_co2}.csv")
    timeseries_stor_discharge_interim.to_csv(f"storDischargeTS_weight_co2_{weight_co2}.csv")
    timeseries_stor_soc_interim.to_csv(f"storSOCTS_weight_co2_{weight_co2}.csv")

pareto_df = pd.DataFrame(pareto_results, columns=['Objective Value', 'Yearly CO2 Limit', 'TAC [€]', 'CO2 Emissions [g]', 'Weight CO2'])
pareto_df.to_csv("Paretofront.csv")


pareto_df = pd.read_csv("Paretofront.csv", index_col=0)

# -------------------------------------------------------------

# 5. Datenausgabe
# 5.1 Darstellung der Pareto-Front

# Umrechnen der CO2-Emissionen in Tonnen und der Kosten (TAC) in Tausend €
pareto_df['CO2 Emissions [t]'] = pareto_df['CO2 Emissions [g]'] / 1000000
pareto_df['TAC [k€]'] = pareto_df['TAC [€]'] / 1000


pareto_df = pareto_df.drop(pareto_df.index[0]).reset_index(drop=True)


pareto_df_grouped = pareto_df.groupby(['TAC [k€]', 'CO2 Emissions [t]'])

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

plt.figure(figsize=(16/2.54, 12/2.54))
plt.scatter(pareto_df['TAC [k€]'], pareto_df['CO2 Emissions [t]'], color=color_raumwaerme)
plt.plot(pareto_df['TAC [k€]'], pareto_df['CO2 Emissions [t]'], linestyle='-', color=color_strom)

for (x, y), group in pareto_df_grouped:
    weights = '; '.join(f'{weight:.2f}'.replace('.', ',') for weight in group['Weight CO2'])
    label = f'$\\alpha_{{CO_2}}$: {weights}'
    if group['Weight CO2'].max() >= 0.8:
        x_offset = -250
    else:
        x_offset = 5
    plt.text(x + 3 + x_offset, y - 3, label, fontsize=10)

plt.xlabel('TAC in Tausend €', fontsize=11)
plt.ylabel('CO$_2$-Emissionen in t', fontsize=11)
plt.gca().xaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.savefig("Abb5_6_Paretofront.pdf")
plt.show()

# -------------------------------------------------------------

# 5.2 Einlesen/Abspeichern der Optimierungsergebnisse

srcSnkSummary = esM.getOptimizationSummary("SourceSinkModel",  ip=2024, outputLevel=1)
display(esM.getOptimizationSummary("SourceSinkModel", ip=2024, outputLevel=1))
tac_srcsnk = srcSnkSummary.xs('TAC', level='Property').sum().sum()
co2_srcsnk = srcSnkSummary.xs(('operation', '[gCO2/h*h/a]'), level=('Property', 'Unit')).loc['CO2 in Atmosphäre'].sum()
display(tac_srcsnk)
display(co2_srcsnk)

convSummary = esM.getOptimizationSummary("ConversionModel", ip=2024, outputLevel=1)
tac_conv = convSummary.xs('TAC', level='Property').sum().sum()
display(esM.getOptimizationSummary("ConversionModel", ip=2024, outputLevel=1))
display(tac_conv)

storSummary = esM.getOptimizationSummary("StorageModel", ip=2024, outputLevel=1)
tac_stor = storSummary.xs('TAC', level='Property').sum().sum()
display(esM.getOptimizationSummary("StorageModel", ip=2024, outputLevel=1))
display(tac_stor)

tac_total = tac_stor + tac_conv + tac_srcsnk
display(tac_total)

timeseries_srcsnk = esM.componentModelingDict[esM.componentNames["Stromkauf von Spotmarkt"]].getOptimalValues("operationVariablesOptimum", ip=2024)
timeseries_srcsnk = timeseries_srcsnk["values"]
timeseries_stor_discharge = esM.componentModelingDict[esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("dischargeOperationVariablesOptimum", ip=2024)
timeseries_stor_discharge = timeseries_stor_discharge["values"]
timeseries_stor_charge = esM.componentModelingDict[esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("chargeOperationVariablesOptimum", ip=2024)
timeseries_stor_charge = timeseries_stor_charge["values"]
timeseries_conv = esM.componentModelingDict[esM.componentNames["BHKW"]].getOptimalValues("operationVariablesOptimum", ip=2024)
timeseries_conv = timeseries_conv["values"]
timeseries_stor_soc = esM.componentModelingDict[esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("stateOfChargeOperationVariablesOptimum", ip=2024)
timeseries_stor_soc = timeseries_stor_soc["values"]

weight_values = np.linspace(0, 0.4, 11)  # Gewichtungen von 0 bis 1 in Schritten von 0.1
weight_co2 = weight_values[1]
dtype_dict = {
    'Component': 'str',
    'Property': 'str',
    'Unit': 'str',
    'FabrikAachen': 'float'
}
srcSnkSummary = pd.read_csv(f"srcSinkSummary_weight_co2_{weight_co2}.csv", dtype=dtype_dict)
convSummary = pd.read_csv(f"convSummary_weight_co2_{weight_co2}.csv", dtype=dtype_dict)
storSummary = pd.read_csv(f"storSummary_weight_co2_{weight_co2}.csv", dtype=dtype_dict)
timeseries_srcsnk = pd.read_csv(f"srcSinkTS_weight_co2_{weight_co2}.csv")
timeseries_conv = pd.read_csv(f"convTS_weight_co2_{weight_co2}.csv")
timeseries_stor_charge = pd.read_csv(f"storChargeTS_weight_co2_{weight_co2}.csv")
timeseries_stor_discharge = pd.read_csv(f"storDischargeTS_weight_co2_{weight_co2}.csv")
timeseries_stor_soc = pd.read_csv(f"storSOCTS_weight_co2_{weight_co2}.csv")
display(srcSnkSummary)
display(convSummary)
display(storSummary)

tac_srcsnk = srcSnkSummary[srcSnkSummary['Property'] == 'TAC']["FabrikAachen"].sum()
co2_srcsnk = srcSnkSummary[srcSnkSummary["Unit"] == '[gCO2/h*h/a]']["FabrikAachen"].sum()
tac_conv = convSummary[convSummary['Property'] == 'TAC']["FabrikAachen"].sum()
tac_stor = storSummary[storSummary['Property'] == 'TAC']["FabrikAachen"].sum()
tac_total = tac_stor + tac_conv + tac_srcsnk
display(tac_total)
display(co2_srcsnk)

# -------------------------------------------------------------
# 5.3 Ausgelegte Leistungen

#cap_pv = srcSnkSummary.loc[("PV", "capacity", "[kW_el]")].sum()#
#cap_bhkw = convSummary.loc[("BHKW", "capacity", "[kW_el]")].sum()
#cap_hp = convSummary.loc[("Wärmepumpe", "capacity", "[kW_th]")].sum()
#cap_tes = storSummary.loc[("Heizungswasserspeicher", "capacity", "[kW_th*h]")].sum()
#cap_lib = storSummary.loc[("Lithium-Ionen-Batterie", "capacity", "[kW_el*h]")].sum()

cap_strom = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Stromkauf von Spotmarkt') & (srcSnkSummary['Property'] == 'capacity') & (srcSnkSummary['Unit'] == '[kW_el]'), 'FabrikAachen'].sum()
cap_pv = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'PV') & (srcSnkSummary['Property'] == 'capacity') & (srcSnkSummary['Unit'] == '[kW_el]'), 'FabrikAachen'].sum()
cap_bhkw = convSummary.loc[(convSummary['Component'] == 'BHKW') & (convSummary['Property'] == 'capacity') & (convSummary['Unit'] == '[kW_el]'), 'FabrikAachen'].sum()
cap_hp = convSummary.loc[(convSummary['Component'] == 'Wärmepumpe') & (convSummary['Property'] == 'capacity') & (convSummary['Unit'] == '[kW_th]'), 'FabrikAachen'].sum()
cap_AKM = convSummary.loc[(convSummary['Component'] == 'Absorptionskältemaschine') & (convSummary['Property'] == 'capacity') & (convSummary['Unit'] == '[kW_th_c]'), 'FabrikAachen'].sum()
cap_KKM = convSummary.loc[(convSummary['Component'] == 'Kompressionskältemaschine') & (convSummary['Property'] == 'capacity') & (convSummary['Unit'] == '[kW_th_c]'), 'FabrikAachen'].sum()
cap_tes = storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') & (storSummary['Property'] == 'capacity') & (storSummary['Unit'] == '[kW_th*h]'), 'FabrikAachen'].sum()
cap_tes_h = storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') & (storSummary['Property'] == 'capacity') & (storSummary['Unit'] == '[kW_th*h]'), 'FabrikAachen'].sum()
cap_tes_c = storSummary.loc[(storSummary['Component'] == 'Kältespeicher') & (storSummary['Property'] == 'capacity') & (storSummary['Unit'] == '[kW_th_c*h]'), 'FabrikAachen'].sum()
cap_lib = storSummary.loc[(storSummary['Component'] == 'Lithium-Ionen-Batterie') & (storSummary['Property'] == 'capacity') & (storSummary['Unit'] == '[kW_el*h]'), 'FabrikAachen'].sum()

categories = ['Netzanschluss-\nleistung\n[kW$_{el}$]', 'PV-Anlage\n[kW$_{el}$]', 'BHKW\n[kW$_{el}$]', 'Wärmepumpe\n[kW$_{th}$]', 'Heizungswassersp.\n[kWh$_{th}$]', 'Kältesp.\n[kWh$_{th}$]', 'LIB\n[kWh$_{el}$]', 'KKM\n[kWh$_{th}$]', 'AKM\n[kWh$_{th}$]']
capacities = [cap_strom, cap_pv, cap_bhkw, cap_hp, cap_tes_h, cap_tes_c , cap_lib, cap_KKM, cap_AKM]

cap_strom_max=5000
cap_pv_max = 2000
cap_bhkw_max = 2000
cap_hp_max = 2000
cap_tes_h_max = 375+2315*3  # 300 + 16 m3
cap_tes_c_max = 700*3       # 300 m3
cap_lib_max = 2000
cap_KKM_max = 300
cap_AKM_max = 420

capacities_max = [cap_strom_max, cap_pv_max, cap_bhkw_max, cap_hp_max, cap_tes_h_max, cap_tes_c_max,
                  cap_lib_max, cap_KKM_max, cap_AKM_max]


def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

y_buffer=max(capacities)*1.25
plt.figure(figsize=(16/2.54, 12/2.54))
bars = plt.bar(categories, capacities, color=[color_strom, gelb_pv, weinrot_bhkw, orange_wp, grün_tes,grün_tes, hellblau_lib, pink_KKM, hellgruen_AKM])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:,.0f}'.replace(",", "."), ha='center', va='bottom', fontsize=11)

for i, max_val in enumerate(capacities_max):
    plt.hlines(y=max_val, xmin=i - 0.4, xmax=i + 0.4, colors='red', linestyles='--', linewidth=2)

plt.ylabel('Leistung in kW / Kapazität in kWh', fontsize=11)
plt.title('Gegebene und optimal ausgebaute Leistungen/Kapazitäten', fontsize=11)
plt.ylim(0, y_buffer)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(f"Kapazitäten_weight_co2_{weight_co2}.pdf", bbox_inches='tight')
plt.show()

# -------------------------------------------------------------
# 5.4 Ausgabe der Kosten


#tac_pv = srcSnkSummary.loc[("PV", "TAC", "[Euro/a]")].sum() # v. a. CAPEX
#tac_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "TAC", "[Euro/a]")].sum() # OPEX
#tac_gas = srcSnkSummary.loc[("Erdgaskauf", "TAC", "[Euro/a]")].sum() # OPEX
#tac_bhkw = convSummary.loc[("BHKW", "TAC", "[Euro/a]")].sum() # OPEX
#tac_hp = convSummary.loc[("Wärmepumpe", "TAC", "[Euro/a]")].sum() # v. a. CAPEX
#tac_tes = storSummary.loc[("Heizungswasserspeicher", "TAC", "[Euro/a]")].sum() # v. a. CAPEX
#tac_lib = storSummary.loc[("Lithium-Ionen-Batterie", "TAC", "[Euro/a]")].sum() # v. a. CAPEX
#tac_co2 = srcSnkSummary.loc[("CO2 in Atmosphäre", "commodCosts", "[Euro/a]")].sum() # nur OPEX

#opex_pv = srcSnkSummary.loc[("PV", "opexCap", "[Euro/a]")].sum()
#opex_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "commodCosts", "[Euro/a]")].sum() - srcSnkSummary.loc[("Stromkauf von Spotmarkt", "commodRevenues", "[Euro/a]")].sum()
#opex_gas = srcSnkSummary.loc[("Erdgaskauf", "commodCosts", "[Euro/a]")].sum()
#opex_bhkw = convSummary.loc[("BHKW", "opexCap", "[Euro/a]")].sum() + convSummary.loc[("BHKW", "opexOp", "[Euro/a]")].sum()
#opex_hp = convSummary.loc[("Wärmepumpe", "opexCap", "[Euro/a]")].sum()
#opex_tes = storSummary.loc[("Heizungswasserspeicher", "opexCap", "[Euro/a]")].sum()
#opex_lib = storSummary.loc[("Lithium-Ionen-Batterie", "opexCap", "[Euro/a]")].sum()
#opex_co2 = srcSnkSummary.loc[("CO2 in Atmosphäre", "commodCosts", "[Euro/a]")].sum()

#capex_pv = srcSnkSummary.loc[("PV", "capexCap", "[Euro/a]")].sum()
#capex_strom = 0
#capex_gas = 0
#capex_bhkw = convSummary.loc[("BHKW", "capexCap", "[Euro/a]")].sum()
#capex_hp = convSummary.loc[("Wärmepumpe", "capexCap", "[Euro/a]")].sum()
#capex_tes = storSummary.loc[("Heizungswasserspeicher", "capexCap", "[Euro/a]")].sum()
#capex_lib = storSummary.loc[("Lithium-Ionen-Batterie", "capexCap", "[Euro/a]")].sum()
#capex_co2 = 0

tac_pv = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'PV') & (srcSnkSummary['Property'] == 'TAC') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_strom = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Stromkauf von Spotmarkt') & (srcSnkSummary['Property'] == 'TAC') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_gas = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Erdgaskauf') & (srcSnkSummary['Property'] == 'TAC') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_bhkw = convSummary.loc[(convSummary['Component'] == 'BHKW') & (convSummary['Property'] == 'TAC') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_hp = convSummary.loc[(convSummary['Component'] == 'Wärmepumpe') & (convSummary['Property'] == 'TAC') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_KKM = convSummary.loc[(convSummary['Component'] == 'Kompressionskältemaschine') & (convSummary['Property'] == 'TAC') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_AKM = convSummary.loc[(convSummary['Component'] == 'Absorptionskältemaschine') & (convSummary['Property'] == 'TAC') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_tes_h = storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') & (storSummary['Property'] == 'TAC') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_tes_c = storSummary.loc[(storSummary['Component'] == 'Kältespeicher') & (storSummary['Property'] == 'TAC') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_lib = storSummary.loc[(storSummary['Component'] == 'Lithium-Ionen-Batterie') & (storSummary['Property'] == 'TAC') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
tac_co2 = srcSnkSummary.loc[(srcSnkSummary['Component'] == "CO2 in Atmosphäre") & (srcSnkSummary['Property'] == "commodCosts") & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()

opex_pv = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'PV') & (srcSnkSummary['Property'] == 'opexCap') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum() + srcSnkSummary.loc[(srcSnkSummary['Component'] == 'PV') & (srcSnkSummary['Property'] == 'commodCosts') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_strom = (srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Stromkauf von Spotmarkt') & (srcSnkSummary['Property'] == 'commodCosts') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum() -
                  srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Stromkauf von Spotmarkt') & (srcSnkSummary['Property'] == 'commodRevenues') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum())
opex_gas = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Erdgaskauf') & (srcSnkSummary['Property'] == 'commodCosts') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_bhkw = (convSummary.loc[(convSummary['Component'] == 'BHKW') & (convSummary['Property'] == 'opexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum() +
                 convSummary.loc[(convSummary['Component'] == 'BHKW') & (convSummary['Property'] == 'opexOp') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum())
opex_hp = convSummary.loc[(convSummary['Component'] == 'Wärmepumpe') & (convSummary['Property'] == 'opexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_KKM = convSummary.loc[(convSummary['Component'] == 'Kompressionskältemaschine') & (convSummary['Property'] == 'opexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum() + convSummary.loc[(convSummary['Component'] == 'Kompressionskältemaschine') & (convSummary['Property'] == 'opexOp') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_AKM = convSummary.loc[(convSummary['Component'] == 'Absorptionskältemaschine') & (convSummary['Property'] == 'opexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum() + convSummary.loc[(convSummary['Component'] == 'Absorptionskältemaschine') & (convSummary['Property'] == 'opexOp') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_tes_h = storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') & (storSummary['Property'] == 'opexCap') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_tes_c = storSummary.loc[(storSummary['Component'] == 'Kältespeicher') & (storSummary['Property'] == 'opexCap') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_lib = storSummary.loc[(storSummary['Component'] == 'Lithium-Ionen-Batterie') & (storSummary['Property'] == 'opexCap') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
opex_co2 = srcSnkSummary.loc[(srcSnkSummary['Component'] == "CO2 in Atmosphäre") & (srcSnkSummary['Property'] == "commodCosts") & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()

capex_pv = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'PV') & (srcSnkSummary['Property'] == 'capexCap') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_strom = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Stromkauf von Spotmarkt') & (srcSnkSummary['Property'] == 'capexCap') & (srcSnkSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_gas = 0
capex_bhkw = convSummary.loc[(convSummary['Component'] == 'BHKW') & (convSummary['Property'] == 'capexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_hp = convSummary.loc[(convSummary['Component'] == 'Wärmepumpe') & (convSummary['Property'] == 'capexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_KKM = convSummary.loc[(convSummary['Component'] == 'Kompressionskältemaschine') & (convSummary['Property'] == 'capexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_AKM = convSummary.loc[(convSummary['Component'] == 'Absorptionskältemaschine') & (convSummary['Property'] == 'capexCap') & (convSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_tes_h = storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') & (storSummary['Property'] == 'capexCap') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_tes_c = storSummary.loc[(storSummary['Component'] == 'Kältespeicher') & (storSummary['Property'] == 'capexCap') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_lib = storSummary.loc[(storSummary['Component'] == 'Lithium-Ionen-Batterie') & (storSummary['Property'] == 'capexCap') & (storSummary['Unit'] == '[Euro/a]'), 'FabrikAachen'].sum()
capex_co2 = 0

# Werte durch 1000 teilen
opex_pv, opex_strom, opex_gas, opex_bhkw, opex_hp,opex_KKM ,opex_AKM , opex_tes_h, opex_tes_c, opex_lib, opex_co2 = [x / 1000 for x in [opex_pv, opex_strom, opex_gas, opex_bhkw, opex_hp, opex_KKM ,opex_AKM , opex_tes_h, opex_tes_c, opex_lib, opex_co2]]
capex_pv, capex_strom, capex_gas, capex_bhkw, capex_hp,capex_KKM,capex_AKM, capex_tes_h, capex_tes_c, capex_lib, capex_co2 = [x / 1000 for x in [capex_pv, capex_strom, capex_gas, capex_bhkw, capex_hp, capex_KKM,capex_AKM, capex_tes_h, capex_tes_c, capex_lib, capex_co2]]
tac_pv, tac_strom, tac_gas, tac_bhkw, tac_hp, tac_KKM, tac_AKM, tac_tes_h,tac_tes_c , tac_lib, tac_co2 = [x / 1000 for x in [tac_pv, tac_strom, tac_gas, tac_bhkw, tac_hp, tac_KKM, tac_AKM, tac_tes_h,tac_tes_c, tac_lib, tac_co2]]

costs = { # €/Jahr
    'Technologie': ['PV-Anlage', 'Stromkauf', 'Erdgaskauf', 'BHKW', 'Wärme-\npumpe', 'KKM', 'AKM', 'Heizungs-\nwassersp.', 'Kältespeicher', 'LIB', "CO$_2$-Preis"],
    'OPEX': [opex_pv, opex_strom, opex_gas, opex_bhkw, opex_hp, opex_KKM ,opex_AKM , opex_tes_h, opex_tes_c, opex_lib, opex_co2],
    'CAPEX': [capex_pv, capex_strom, capex_gas, capex_bhkw, capex_hp, capex_KKM,capex_AKM, capex_tes_h, capex_tes_c, capex_lib, capex_co2],
    'TAC': [tac_pv, tac_strom, tac_gas, tac_bhkw, tac_hp, tac_KKM, tac_AKM, tac_tes_h,tac_tes_c, tac_lib, tac_co2]
}

df_costs = pd.DataFrame(costs)

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

fig, ax = plt.subplots(figsize=(16/2.54, 12/2.54))
df_costs.set_index('Technologie')[['OPEX', 'CAPEX']].plot(kind='bar', stacked=True, ax=ax, color= [color_strom, color_raumwaerme])

for i, j in enumerate(df_costs['TAC']):
    tac_formatted = f'{j:,.1f}'.replace(',', 'TEMP').replace('.', ',').replace('TEMP', '.')
    ax.text(i, j + 20, tac_formatted, ha='center', fontsize=11) # i is the x-coordinate where the text will be placed, corresponding to the position of the bar. j + 50 is the y-coordinate where the text will be placed, slightly above the top of the bar. f'{j}' is the text that will be displayed, which is the total value from the TAC column.

ax.set_xlabel("")
plt.ylabel('Kosten in Tausend €', fontsize=11)
max_tac = df_costs['TAC'].max()
plt.ylim(0, max_tac * 1.15)  # 15 % Puffer nach oben
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.legend(fontsize=11, loc="upper center")
plt.savefig(f"Kosten_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()

# -------------------------------------------------------------
# 5.5 Ergebnisse Betriebsoptimierung


#operation_strombedarf = srcSnkSummary.loc[("Strombedarf", "operation", "[kW_el*h/a]")].sum()
#operation_heizungswasserbedarf = srcSnkSummary.loc[("Heizungswasserbedarf", "operation", "[kW_th*h/a]")].sum()
#operation_pv = srcSnkSummary.loc[("PV", "operation", "[kW_el*h/a]")].sum()
#operation_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "operation", "[kW_el*h/a]")].sum()
#operation_gas = srcSnkSummary.loc[("Erdgaskauf", "operation", "[kW_CH4,LHV*h/a]")].sum()
#operation_bhkw = convSummary.loc[("BHKW", "operation", "[kW_el*h/a]")].sum()
#operation_hp = convSummary.loc[("Wärmepumpe", "operation", "[kW_th*h/a]")].sum()
#operation_tes = (storSummary.loc[("Heizungswasserspeicher", "operationCharge", "[kW_th*h/a]")].sum() + storSummary.loc[("Heizungswasserspeicher", "operationDischarge", "[kW_th*h/a]")].sum())/2 # Mittelwert
#operation_lib = (storSummary.loc[("Lithium-Ionen-Batterie", "operationCharge", "[kW_el*h/a]")].sum() + storSummary.loc[("Lithium-Ionen-Batterie", "operationDischarge", "[kW_el*h/a]")].sum())/2 # Mittelwert

operation_strombedarf = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Strombedarf') &
                                              (srcSnkSummary['Property'] == 'operation') &
                                              (srcSnkSummary['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum()

operation_heizungswasserbedarf = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Heizungswasserbedarf') &
                                                       (srcSnkSummary['Property'] == 'operation') &
                                                       (srcSnkSummary['Unit'] == '[kW_th*h/a]'), 'FabrikAachen'].sum()

operation_kältebedarf = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Kältebedarf') &
                                                       (srcSnkSummary['Property'] == 'operation') &
                                                       (srcSnkSummary['Unit'] == '[kW_th_c*h/a]'), 'FabrikAachen'].sum()

operation_pv = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'PV') &
                                     (srcSnkSummary['Property'] == 'operation') &
                                     (srcSnkSummary['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum()

operation_strom = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Stromkauf von Spotmarkt') &
                                        (srcSnkSummary['Property'] == 'operation') &
                                        (srcSnkSummary['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum()

operation_gas = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Erdgaskauf') &
                                      (srcSnkSummary['Property'] == 'operation') &
                                      (srcSnkSummary['Unit'] == '[kW_CH4,LHV*h/a]'), 'FabrikAachen'].sum()

operation_bhkw = convSummary.loc[(convSummary['Component'] == 'BHKW') &
                                     (convSummary['Property'] == 'operation') &
                                     (convSummary['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum()

operation_AKM = convSummary.loc[(convSummary['Component'] == 'Absorptionskältemaschine') &
                                     (convSummary['Property'] == 'operation') &
                                     (convSummary['Unit'] == '[kW_th_c*h/a]'), 'FabrikAachen'].sum()

operation_KKM = convSummary.loc[(convSummary['Component'] == 'Kompressionskältemaschine') &
                                     (convSummary['Property'] == 'operation') &
                                     (convSummary['Unit'] == '[kW_th_c*h/a]'), 'FabrikAachen'].sum()

operation_hp = convSummary.loc[(convSummary['Component'] == 'Wärmepumpe') &
                                   (convSummary['Property'] == 'operation') &
                                   (convSummary['Unit'] == '[kW_th*h/a]'), 'FabrikAachen'].sum()
'''
operation_tes_h = (storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') &
                                     (storSummary['Property'] == 'operationCharge') &
                                     (storSummary['Unit'] == '[kW_th*h/a]'), 'FabrikAachen'].sum() +
                     storSummary.loc[(storSummary['Component'] == 'Heisswasserspeicher') &
                                     (storSummary['Property'] == 'operationDischarge') &
                                     (storSummary['Unit'] == '[kW_th*h/a]'), 'FabrikAachen'].sum()) / 2  # Mittelwert

operation_tes_c = (storSummary.loc[(storSummary['Component'] == 'Kältespeicher') &
                                     (storSummary['Property'] == 'operationCharge') &
                                     (storSummary['Unit'] == '[kW_th_c*h/a]'), 'FabrikAachen'].sum() +
                     storSummary.loc[(storSummary['Component'] == 'Heisswasserspeicher') &
                                     (storSummary['Property'] == 'operationDischarge') &
                                     (storSummary['Unit'] == '[kW_th_c*h/a]'), 'FabrikAachen'].sum()) / 2  # Mittelwert
'''
operation_tes_h = (storSummary.loc[(storSummary['Component'] == 'Heizungswasserspeicher') &
                                     (storSummary['Property'] == 'operationDischarge') &
                                     (storSummary['Unit'] == '[kW_th*h/a]'), 'FabrikAachen'].sum())


operation_tes_c = storSummary.loc[(storSummary['Component'] == 'Kältespeicher') &
                                     (storSummary['Property'] == 'operationDischarge') &
                                     (storSummary['Unit'] == '[kW_th_c*h/a]'), 'FabrikAachen'].sum()


operation_lib = (storSummary.loc[(storSummary['Component'] == 'Lithium-Ionen-Batterie') &
                                     (storSummary['Property'] == 'operationCharge') &
                                     (storSummary['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum() +
                     storSummary.loc[(storSummary['Component'] == 'Lithium-Ionen-Batterie') &
                                     (storSummary['Property'] == 'operationDischarge') &
                                     (storSummary['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum()) / 2  # Mittelwert

jahressummen = [x / 1000000 for x in [operation_pv, operation_strom, operation_gas, operation_bhkw, operation_hp, operation_tes_h,operation_tes_c,operation_AKM,operation_KKM,  operation_lib]]
jahresbedarfe = [x / 1000000 for x in [operation_strombedarf, operation_heizungswasserbedarf, operation_kältebedarf]]

categories2 = ['PV-Anlage [GWh$_{el}$]', "Strombezug [GWh$_{el}$]", "Gasbezug [GWh$_{CH_4}$]", 'BHKW [GWh$_{el}$]', 'Wärmepumpe [GWh$_{th}$]', 'Heizungs-\nwassersp. [GWh$_{th}$]','Kältespeicher [GWh$_{th}$]','AKM [GWh$_{th_c}$]', 'KKM [GWh$_{th_c}$]', 'LIB [GWh$_{el}$]']
categories3 = ['Strom [GWh$_{el}$]', '  Raumwärme [GWh$_{th}$]', 'Kälte [GWh$_{th_c}$]']

formatter = FuncFormatter(lambda x, _: f'{x:.1f}'.replace('.', ','))
y_buffer=max(jahressummen)*1.25

fig1, ax1 = plt.subplots(figsize=(16 / 2.54, 16 / 2.54))
bars1 = ax1.barh(categories2, jahressummen, color=[gelb_pv, blau_stromkauf, lila_gas, weinrot_bhkw, orange_wp, grün_tes, grün_tes, hellgruen_AKM, pink_KKM, hellblau_lib])
for bar in bars1:
    xval = bar.get_width()
    xval_formatted = f'{xval:.2f}'.replace('.', ',')
    ax1.text(xval + 0.5, bar.get_y() + bar.get_height() / 2, xval_formatted, ha='left', va='center', fontsize=11)
ax1.set_xlabel('Jährliche Energie pro Modellkomponente in GWh', fontsize=11)
ax1.set_yticks(ax1.get_yticks())
ax1.set_yticklabels(categories2, rotation=0, fontsize=11)
ax1.xaxis.set_major_formatter(formatter)
ax1.set_xlim(0, y_buffer)
plt.tight_layout()
plt.savefig(f"operation_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(16 / 2.54, 5.6 / 2.54))
bars2 = ax2.barh(categories3, jahresbedarfe, color=[color_strom, color_raumwaerme, color_kälte])
for bar in bars2:
    xval = bar.get_width()
    xval_formatted = f'{xval:.2f}'.replace('.', ',')
    ax2.text(xval + 0.1, bar.get_y() + bar.get_height() / 2, xval_formatted, ha='left', va='center', fontsize=11)
ax2.set_xlabel('Jährlicher Nutzenergiebedarf in GWh', fontsize=11)
ax2.set_yticks(ax2.get_yticks())
ax2.set_yticklabels(categories3, rotation=0, fontsize=11)
ax2.xaxis.set_major_formatter(formatter)
ax2.set_xlim(0, 19)

plt.tight_layout()
plt.savefig(f"operation_weight_co2_{weight_co2}_2.pdf", bbox_inches="tight")
plt.show()

# -------------------------------------------------------------

print(srcSnkSummary.index)
print(srcSnkSummary.index.names)
print(srcSnkSummary.head())

# 6. Berechnung der KPIs

# Anteil des Selbsterzeugten Stroms/ Autakiequote=verbrauchter Solarstrom/gesamter Stromverbrauch

selbsterzeugungsanteil=operation_pv/operation_strombedarf

print(f'Der Selbsterzeugungsanteil durch PV beträgt {selbsterzeugungsanteil:.2%}')

#PV-Nutzungsquote Eigenverbrauchsquote
#einspeisung_pv = srcSnkSummary.loc[("Überschussstrom in Netz", "operation", "[kW_el*h/a]")].sum()

einspeisung_pv = srcSnkSummary.loc[(srcSnkSummary['Component'] == 'Überschussstrom in Netz') &
    (srcSnkSummary['Property'] == 'operation') &
    (srcSnkSummary['Unit'] == '[kW_el*h/a]')]['FabrikAachen'].sum()

print(f"PV-Einspeisung ins Netz: {einspeisung_pv / 1e6:.2f} GWh")

eigenverbrauchsquote=(operation_pv-einspeisung_pv)/operation_pv
print(f'Die PV-Nutzungsquote  PV beträgt {eigenverbrauchsquote:.2%}')


kpi_labels = ['Selbsterzeugungsanteil', 'PV-Eigenverbrauchsquote']
kpi_values = [selbsterzeugungsanteil, eigenverbrauchsquote]

plt.figure(figsize=(6, 4))
bars = plt.bar(kpi_labels, kpi_values, color=['green', 'blue'])
plt.ylabel('Anteil [%]')
plt.ylim(0, 1)
plt.title('PV-Kennzahlen')

# Prozentwerte direkt auf die Balken schreiben
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.1%}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------

# 7. Piecharts Energiebedarfe

# 7.1 Pie Chart zur Jahreskälteleistung


# Kältemengen aus Optimierungsergebnissen (in kWh)
kälte_AKM = operation_AKM
kälte_KKM = operation_KKM

### Pie Charts
labels = ['AKM', 'KKM']
sizes = [kälte_AKM, kälte_KKM]
colors = [hellgruen_AKM, pink_KKM]  # aus deiner Farbdefinition
sizes_gwh = [x / 1e6 for x in sizes]
total = sum(sizes_gwh)
labels_with_pct = [f"{label} ({size / total * 100:.1f}%)" for label, size in zip(labels, sizes_gwh)]

plt.figure(figsize=(8, 8))
wedges, texts = plt.pie(
    sizes_gwh,
    labels=None,
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11},
)

plt.legend(wedges, labels_with_pct, title="Komponenten", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=13)
plt.title('Jahreskälteleistung: AKM vs. KKM', fontsize=13)
plt.axis('equal')  # Kreis statt Oval
plt.tight_layout()
plt.show()
# -------------------------------------------------------------

# 7.2  Strommengen in kWh
strom_pv = operation_pv
strom_bhkw = operation_bhkw
strom_bezug = operation_strom

labels = ['PV-Erzeugung', 'BHKW-Erzeugung', 'Strombezug Spotmarkt']
sizes = [strom_pv, strom_bhkw, strom_bezug]
colors = [gelb_pv, weinrot_bhkw, blau_stromkauf]

sizes_gwh = [x / 1e6 for x in sizes]
total = sum(sizes_gwh)
labels_with_pct = [f"{label} ({size / total * 100:.1f}%)" for label, size in zip(labels, sizes_gwh)]

plt.figure(figsize=(8, 8))
wedges, texts = plt.pie(
    sizes_gwh,
    labels=None,
    startangle=90,
    colors=colors,
    textprops={'fontsize': 11},
)

plt.legend(wedges, labels_with_pct, title="Komponenten", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=13)
plt.title('Stromerzeugung und -bezug in 2024', fontsize=13)
plt.axis('equal')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------

# 7.3 Pie Chart zur Aufteilung TAC

labels = ['PV-Anlage', 'Stromkauf', 'Erdgaskauf', 'BHKW', 'Wärme-\npumpe', 'KKM', 'AKM', 'Heizungs-\nwassersp.', 'Kältespeicher', 'LIB', "CO$_2$-Preis"]
sizes = [tac_pv, tac_strom, tac_gas, tac_bhkw, tac_hp, tac_KKM, tac_AKM, tac_tes_h,tac_tes_c, tac_lib, tac_co2]
colors = [gelb_pv, color_strom, lila_gas, weinrot_bhkw, hellgruen_AKM, pink_KKM, grün_tes, grün_tes, hellblau_lib, color_raumwaerme]
explode = [0, 0, 0, 0.05, 0.1, 0.1, 0.15, 0.1, 0, 0, 0]

sizes_gwh = [x / 1e6 for x in sizes]
total = sum(sizes_gwh)
labels_with_pct = [f"{label} ({size / total * 100:.1f}%)" for label, size in zip(labels, sizes_gwh)]

plt.figure(figsize=(8, 8))
wedges, texts = plt.pie(
    sizes_gwh,
    labels=None,
    startangle=90,
    colors=colors,
    explode=explode,
    textprops={'fontsize': 12},
    pctdistance=1.2,
)

plt.legend(wedges, labels_with_pct, title="Komponenten", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=12, title_fontsize=13)
plt.title('Verteilung der TAC in 2024', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.show()
# -------------------------------------------------------------

# Zukauf von Strom
fig, ax=fn.standardIO.plotOperation(
    esM,
    compName='Stromkauf von Spotmarkt',
    loc='FabrikAachen',
    ip=2024,
    variableName='operationVariablesOptimum',
    xlabel='Stunde',
    ylabel='Strombezug in kW',
    figsize=(12, 4),
    color='blue',
    fontsize=12
)
ax.set_title("Stromkauf von Spotmarkt")
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Verteilung CO2 Emissionen
co2_gesamt = co2_srcsnk
co2_gas = operation_gas * co2_faktor_gas
co2_strom = co2_gesamt - co2_gas
co2_gesamt_gerundet = int(co2_gesamt/1000000)

'''
labels = ['CO$_2$-Emissionen\ndurch BHKW', 'CO$_2$-Emissionen\ndurch eingekauften Strom']
sizes = [co2_gas, co2_strom]
colors = [color_raumwaerme, color_strom]

fig, ax = plt.subplots(figsize=(16/2.54, 12/2.54))
wedges, texts, autotexts = ax.pie(sizes, colors=colors, autopct='%1.0f %%', startangle=0, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
ax.axis('equal')
plt.legend(wedges, labels, loc="lower right", fontsize=11)

plt.title(f'Gesamte CO$_2$-Emissionen: {co2_gesamt_gerundet} t', fontsize=13)

#plt.savefig(f"THGE_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()
'''
#-------------------------- Heatmaps--------------------------------

# 8. Heatmaps

# 8.1 Funktionsdefinition


def plot_heatmap_komponente(csv_path, komponente, cmap='plasma', save=False, output_name=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    row = df[df["Unnamed: 0"] == komponente]

    werte_raw = row.iloc[0, 2:]
    werte = pd.to_numeric(werte_raw, errors='coerce')

    matrix = werte.to_numpy().reshape((366, 24))

    plt.figure(figsize=(16/2.54, 12/2.54))
    plt.imshow(matrix.T, aspect='auto', origin='lower', cmap=custom_cmap_rot)
    plt.xlabel("Tag des Jahres")
    plt.ylabel("Stunde des Tages")
    plt.title(f"{komponente}")
    plt.colorbar(label="Leistung [kW]")
    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf")
    plt.show()

def plot_heatmap_speicher(csv_path, komponente, cmap='plasma', save=False, output_name=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    row = df[df["Unnamed: 0"] == komponente]

    werte_raw = row.iloc[0, 2:]
    werte = pd.to_numeric(werte_raw, errors='coerce')

    matrix = werte.to_numpy().reshape((366, 24))

    plt.figure(figsize=(16/2.54, 12/2.54))
    plt.imshow(matrix.T, aspect='auto', origin='lower', cmap=custom_cmap_rot)
    plt.xlabel("Tag des Jahres")
    plt.ylabel("Stunde des Tages")
    plt.title(f"{komponente}")
    plt.colorbar(label="SOC [kWh]")
    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf")
    plt.show()

def plot_heatmap_CO2(csv_path, komponente, cmap='plasma', save=False, output_name=None):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    row = df[df["Unnamed: 0"] == komponente]

    werte_raw = row.iloc[0, 2:]
    werte = pd.to_numeric(werte_raw, errors='coerce')

    matrix = werte.to_numpy().reshape((366, 24))

    plt.figure(figsize=(16/2.54, 12/2.54))
    plt.imshow(matrix.T, aspect='auto', origin='lower', cmap=custom_cmap_rot)
    plt.xlabel("Tag des Jahres")
    plt.ylabel("Stunde des Tages")
    plt.title(f"{komponente}")
    plt.colorbar(label="CO2 in Atmosphäre [g]")
    plt.tight_layout()
    plt.savefig(f"{output_name}.pdf")
    plt.show()

# -------------------------------------------------------------

# 8.2 Funktionsaufruf

plot_heatmap_komponente(f"convTS_weight_co2_{weight_co2}.csv", "Absorptionskältemaschine", cmap="plasma", save=True)
plot_heatmap_komponente(f"convTS_weight_co2_{weight_co2}.csv", "Kompressionskältemaschine", cmap="plasma", save=True)
plot_heatmap_komponente(f"convTS_weight_co2_{weight_co2}.csv", "BHKW", cmap="plasma", save=True)
plot_heatmap_komponente(f"convTS_weight_co2_{weight_co2}.csv", "Wärmepumpe", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "Strombedarf", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "Kältebedarf", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "Heizungswasserbedarf", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "Stromkauf von Spotmarkt", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "Erdgaskauf", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "Überschussstrom in Netz", cmap="plasma", save=True)
plot_heatmap_komponente(f"srcSinkTS_weight_co2_{weight_co2}.csv", "PV", cmap="plasma", save=True)

plot_heatmap_CO2(f"srcSinkTS_weight_co2_{weight_co2}.csv", "CO2 in Atmosphäre", cmap="plasma", save=True)
plot_heatmap_speicher(f"storSOCTS_weight_co2_{weight_co2}.csv", "Kältespeicher", cmap="plasma", save=True)
plot_heatmap_speicher(f"storSOCTS_weight_co2_{weight_co2}.csv", "Heizungswasserspeicher", cmap="plasma", save=True)
plot_heatmap_speicher(f"storSOCTS_weight_co2_{weight_co2}.csv", "Lithium-Ionen-Batterie", cmap="plasma", save=True)

# -------------------------------------------------------------


# 9. Plots für Gewichtungsparameter

#-------------------------- Plots für Gewichtungsparameter--------------------------------

# 9.1 Histogram Auslegungen alle weight values


capacity_results = {
    "Netzanschlussleistung": [],
    "PV": [],
    "BHKW": [],
    "Wärmepumpe": [],
    "AKM": [],
    "KKM": [],
    "Heizungswasserspeicher": [],
    "Kältespeicher": [],
    "LIB": []
}


# Iteriere über alle weight_values
for w in weight_values:
    # Lade die jeweiligen Dateien
    srcSnkSummary_w = pd.read_csv(f"srcSinkSummary_weight_co2_{w}.csv")
    convSummary_w = pd.read_csv(f"convSummary_weight_co2_{w}.csv")
    storSummary_w = pd.read_csv(f"storSummary_weight_co2_{w}.csv")

    # Extrahiere Kapazitäten
    cap_strom_w = srcSnkSummary_w.loc[(srcSnkSummary_w['Component'] == 'Stromkauf von Spotmarkt') & (srcSnkSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_pv_w = srcSnkSummary_w.loc[(srcSnkSummary_w['Component'] == 'PV') & (srcSnkSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_bhkw_w = convSummary_w.loc[(convSummary_w['Component'] == 'BHKW') & (convSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_hp_w = convSummary_w.loc[(convSummary_w['Component'] == 'Wärmepumpe') & (convSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_AKM_w = convSummary_w.loc[(convSummary_w['Component'] == 'Absorptionskältemaschine') & (convSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_KKM_w = convSummary_w.loc[(convSummary_w['Component'] == 'Kompressionskältemaschine') & (convSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_tes_w = storSummary_w.loc[(storSummary_w['Component'] == 'Heizungswasserspeicher') & (storSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_kalt_w = storSummary_w.loc[(storSummary_w['Component'] == 'Kältespeicher') & (storSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()
    cap_lib_w = storSummary_w.loc[(storSummary_w['Component'] == 'Lithium-Ionen-Batterie') & (storSummary_w['Property'] == 'capacity'), 'FabrikAachen'].sum()

    # In Dictionary speichern
    capacity_results["Netzanschlussleistung"].append(cap_strom_w)
    capacity_results["PV"].append(cap_pv_w)
    capacity_results["BHKW"].append(cap_bhkw_w)
    capacity_results["Wärmepumpe"].append(cap_hp_w)
    capacity_results["AKM"].append(cap_AKM_w)
    capacity_results["KKM"].append(cap_KKM_w)
    capacity_results["Heizungswasserspeicher"].append(cap_tes_w)
    capacity_results["Kältespeicher"].append(cap_kalt_w)
    capacity_results["LIB"].append(cap_lib_w)

# DataFrame zum Plotten erstellen
df_caps = pd.DataFrame(capacity_results, index=weight_values)

# Balkendiagramm (Histogram-artig) erzeugen
fig, ax = plt.subplots(figsize=(18/2.54, 10/2.54))
colors_caps = [farben_komponenten[comp] for comp in df_caps.columns]
df_caps.plot(kind='bar', ax=ax, width=0.9, color=colors_caps)


plt.xlabel("CO2-Gewichtung")
plt.ylabel("Kapazität / Leistung")
plt.title("Auslegung der Komponenten über alle Gewichtungsfaktoren")
plt.xticks(ticks=np.arange(len(weight_values)), labels=[f"{w:.2f}" for w in weight_values], rotation=45)
#plt.legend(loc='upper left', fontsize=9)
plt.legend(
    fontsize=9,
    title="Komponenten",
    title_fontsize=9,
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)
plt.tight_layout()
plt.savefig("Histogram_Auslegungen_alle_weight_values.pdf", bbox_inches="tight")
plt.show()

# -------------------------------------------------------------

# 9.2 Conversion Kapazitäten über Gewichtungsfaktoren


conversion_caps = {
    "BHKW": [],
    "Wärmepumpe": [],
    "Absorptionskältemaschine": [],
    "Kompressionskältemaschine": []
}

for w in weight_values:
    convSummary_w = pd.read_csv(f"convSummary_weight_co2_{w}.csv")

    # Für jede Conversion-Komponente extrahieren
    for comp in conversion_caps.keys():
        val = convSummary_w.loc[
            (convSummary_w["Component"] == comp) &
            (convSummary_w["Property"] == "capacity"),
            "FabrikAachen"
        ].sum()
        conversion_caps[comp].append(val)

# DataFrame
df_conv = pd.DataFrame(conversion_caps, index=weight_values)

# Plot
plt.figure(figsize=(16/2.54,10/2.54))
for comp in df_conv.columns:
    plt.plot(
        df_conv.index,
        df_conv[comp],
        marker="o",
        color=farben_komponenten[comp],   # hier Farbe nach Dictionary
        label=comp
    )

plt.xlabel("CO$_2$-Gewichtung", fontsize=11)
plt.ylabel("Kapazität [kW]", fontsize=11)
plt.title("Conversion-Komponenten über Gewichtungsparameter", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("Conversion_Kapazitäten_über_Gewichtungsfaktoren.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 9.3 Storage Kapazitäten über Gewichtungsfaktoren

# Dictionary für Storage
storage_caps = {
    "Heizungswasserspeicher": [],
    "Kältespeicher": [],
    "Lithium-Ionen-Batterie": []
}

for w in weight_values:
    storSummary_w = pd.read_csv(f"storSummary_weight_co2_{w}.csv")

    for comp in storage_caps.keys():
        val = storSummary_w.loc[
            (storSummary_w["Component"] == comp) &
            (storSummary_w["Property"] == "capacity"),
            "FabrikAachen"
        ].sum()
        storage_caps[comp].append(val)

# DataFrame
df_stor = pd.DataFrame(storage_caps, index=weight_values)

# Plot
plt.figure(figsize=(16/2.54,10/2.54))
for comp in df_stor.columns:
    plt.plot(
        df_stor.index,
        df_stor[comp],
        marker="o",
        color=farben_komponenten[comp],   # hier Farbe nach Dictionary
        label=comp
    )

plt.xlabel("CO$_2$-Gewichtung", fontsize=11)
plt.ylabel("Kapazität [kW bzw. kWh]", fontsize=11)
plt.title("Speicher-Komponenten über Gewichtungsparameter", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("Storage_Kapazitäten_über_Gewichtungsfaktoren.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 9.4 SourcesSinks Jahresenergie über Gewichtungsfaktoren

# Dictionary für alle relevanten Sources/Sinks Jahresenergiemengen
srcsnk_energy = {
    "PV": [],
    "Stromkauf": [],
    "Erdgaskauf": [],
}

for w in weight_values:
    srcSnkSummary_w = pd.read_csv(f"srcSinkSummary_weight_co2_{w}.csv")

    # PV [kWh_el]
    pv = srcSnkSummary_w.loc[
        (srcSnkSummary_w["Component"] == "PV") &
        (srcSnkSummary_w["Property"] == "operation") &
        (srcSnkSummary_w["Unit"] == "[kW_el*h/a]"),
        "FabrikAachen"
    ].sum()

    # Stromkauf [kWh_el]
    strom = srcSnkSummary_w.loc[
        (srcSnkSummary_w["Component"] == "Stromkauf von Spotmarkt") &
        (srcSnkSummary_w["Property"] == "operation") &
        (srcSnkSummary_w["Unit"] == "[kW_el*h/a]"),
        "FabrikAachen"
    ].sum()

    # Erdgaskauf [kWh_CH4]
    gas = srcSnkSummary_w.loc[
        (srcSnkSummary_w["Component"] == "Erdgaskauf") &
        (srcSnkSummary_w["Property"] == "operation") &
        (srcSnkSummary_w["Unit"] == "[kW_CH4,LHV*h/a]"),
        "FabrikAachen"
    ].sum()


    # In GWh umrechnen
    srcsnk_energy["PV"].append(pv / 1e6)
    srcsnk_energy["Stromkauf"].append(strom / 1e6)
    srcsnk_energy["Erdgaskauf"].append(gas / 1e6)

# DataFrame
df_srcsnk_energy = pd.DataFrame(srcsnk_energy, index=weight_values)

# Plot
plt.figure(figsize=(18/2.54,10/2.54))
for comp in df_srcsnk_energy.columns:
    plt.plot(
        df_srcsnk_energy.index,
        df_srcsnk_energy[comp],
        marker="o",
        color=farben_komponenten[comp],  # hier Farbe nach Dictionary
        label=comp
    )


plt.xlabel("CO$_2$-Gewichtung", fontsize=11)
plt.ylabel("Jahresenergie [GWh]", fontsize=11)
plt.title("Sources/Sinks Jahresenergie über Gewichtungsparameter", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("SourcesSinks_Jahresenergie_über_Gewichtungsfaktoren.pdf", bbox_inches="tight")
plt.show()
# ------------------------------------------------------------

# 9.5 Conversion Jahresenergie über Gewichtungsfaktoren

conv_energy = {
    "BHKW": [],
    "Wärmepumpe": [],
    "Absorptionskältemaschine": [],
    "Kompressionskältemaschine": []
}

for w in weight_values:
    convSummary_w = pd.read_csv(f"convSummary_weight_co2_{w}.csv")

    # BHKW [kWh_el]
    bhkw = convSummary_w.loc[
        (convSummary_w["Component"] == "BHKW") &
        (convSummary_w["Property"] == "operation") &
        (convSummary_w["Unit"] == "[kW_el*h/a]"),
        "FabrikAachen"
    ].sum()

    # Wärmepumpe [kWh_th]
    hp = convSummary_w.loc[
        (convSummary_w["Component"] == "Wärmepumpe") &
        (convSummary_w["Property"] == "operation") &
        (convSummary_w["Unit"] == "[kW_th*h/a]"),
        "FabrikAachen"
    ].sum()

    # AKM [kWh_th_c]
    akm = convSummary_w.loc[
        (convSummary_w["Component"] == "Absorptionskältemaschine") &
        (convSummary_w["Property"] == "operation") &
        (convSummary_w["Unit"] == "[kW_th_c*h/a]"),
        "FabrikAachen"
    ].sum()

    # KKM [kWh_th_c]
    kkm = convSummary_w.loc[
        (convSummary_w["Component"] == "Kompressionskältemaschine") &
        (convSummary_w["Property"] == "operation") &
        (convSummary_w["Unit"] == "[kW_th_c*h/a]"),
        "FabrikAachen"
    ].sum()

    # In GWh umrechnen
    conv_energy["BHKW"].append(bhkw / 1e6)
    conv_energy["Wärmepumpe"].append(hp / 1e6)
    conv_energy["Absorptionskältemaschine"].append(akm / 1e6)
    conv_energy["Kompressionskältemaschine"].append(kkm / 1e6)

# DataFrame
df_conv_energy = pd.DataFrame(conv_energy, index=weight_values)

# Plot
plt.figure(figsize=(18/2.54,10/2.54))
for comp in df_conv_energy.columns:
    plt.plot(
        df_conv_energy.index,
        df_conv_energy[comp],
        marker="o",
        color=farben_komponenten[comp],  # hier Farbe nach Dictionary
        label=comp
    )

plt.xlabel("CO$_2$-Gewichtung", fontsize=11)
plt.ylabel("Jahresenergie [GWh]", fontsize=11)
plt.title("Conversion-Komponenten Jahresenergie über Gewichtungsparameter", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("Conversion_Jahresenergie_über_Gewichtungsfaktoren.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 9.6 Storage Jahresenergie Discharge über Gewichtungsfaktoren

# Dictionary für Storage-Komponenten Jahresenergiemengen (nur Discharge)

storage_energy = {
    "Heizungswasserspeicher": [],
    "Kältespeicher": [],
    "LIB": []
}

for w in weight_values:
    storSummary_w = pd.read_csv(f"storSummary_weight_co2_{w}.csv")

    # Heizungswasserspeicher nur Discharge [kWh_th]
    tes_h = storSummary_w.loc[
        (storSummary_w["Component"] == "Heizungswasserspeicher") &
        (storSummary_w["Property"] == "operationDischarge") &
        (storSummary_w["Unit"] == "[kW_th*h/a]"),
        "FabrikAachen"
    ].sum()

    # Kältespeicher nur Discharge [kWh_th_c]
    tes_c = storSummary_w.loc[
        (storSummary_w["Component"] == "Kältespeicher") &
        (storSummary_w["Property"] == "operationDischarge") &
        (storSummary_w["Unit"] == "[kW_th_c*h/a]"),
        "FabrikAachen"
    ].sum()

    # LIB nur Discharge [kWh_el]
    lib = storSummary_w.loc[
        (storSummary_w["Component"] == "Lithium-Ionen-Batterie") &
        (storSummary_w["Property"] == "operationDischarge") &
        (storSummary_w["Unit"] == "[kW_el*h/a]"),
        "FabrikAachen"
    ].sum()

    # In GWh umrechnen
    storage_energy["Heizungswasserspeicher"].append(tes_h / 1e6)
    storage_energy["Kältespeicher"].append(tes_c / 1e6)
    storage_energy["LIB"].append(lib / 1e6)

# DataFrame
df_storage_energy = pd.DataFrame(storage_energy, index=weight_values)

# Plot
plt.figure(figsize=(18/2.54, 10/2.54))
for comp in df_storage_energy.columns:
    plt.plot(
        df_storage_energy.index,
        df_storage_energy[comp],
        marker="o",
        color=farben_komponenten[comp],  # hier Farbe nach Dictionary
        label=comp
    )

plt.xlabel("CO$_2$-Gewichtung", fontsize=11)
plt.ylabel("Jahresenergie [GWh]", fontsize=11)
plt.title("Storage-Komponenten Jahresenergie (nur Discharge) über Gewichtungsparameter", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("Storage_Jahresenergie_Discharge_über_Gewichtungsfaktoren.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 9.7 Heizungswasserspeicher Jahresenergie Discharge

plt.figure(figsize=(18/2.54, 10/2.54))

plt.plot(
    df_storage_energy.index,
    df_storage_energy["Heizungswasserspeicher"],
    marker="o",
    color=grün_tes,
    label="Heizungswasserspeicher"
)

plt.xlabel("CO$_2$-Gewichtung", fontsize=11)
plt.ylabel("Jahresenergie [GWh]", fontsize=11)
plt.title("Jahresenergie Heizungswasserspeicher (nur Discharge) über CO$_2$-Gewichtung", fontsize=12)


plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("Heizungswasserspeicher_Jahresenergie_Discharge.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 9.8 LIB Kältespeicher Jahresenergie Discharge

plt.figure(figsize=(18/2.54, 10/2.54))

# Kältespeicher
plt.plot(
    df_storage_energy.index,
    df_storage_energy["Kältespeicher"],
    marker="s",
    label="Kältespeicher"
)

# LIB
plt.plot(
    df_storage_energy.index,
    df_storage_energy["LIB"],
    marker="o",
    color=farben_komponenten[comp],  # hier Farbe nach Dictionary
    label="Lithium-Ionen-Batterie"
)

plt.xlabel("CO$_2$-Gewichtung", fontsize=13)
plt.ylabel("Jahresenergie [GWh]", fontsize=13)
plt.title("Jahresenergie Kältespeicher & Lithium-Ionen-Batterie (nur Discharge)", fontsize=14, pad=15)

plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(weight_values, [f"{w:.2f}" for w in weight_values], rotation=45)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("LIB_Kältespeicher_Jahresenergie_Discharge.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 10. Line Plots

# 10.1 Optimierungsergebnisse strukturiert einlesen

timeseries_srcsnk = esM.componentModelingDict[esM.componentNames["Stromkauf von Spotmarkt"]].getOptimalValues("operationVariablesOptimum", ip=2024)
timeseries_srcsnk = timeseries_srcsnk["values"]
timeseries_stor_discharge = esM.componentModelingDict[esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("dischargeOperationVariablesOptimum", ip=2024)
timeseries_stor_discharge = timeseries_stor_discharge["values"]
timeseries_stor_charge = esM.componentModelingDict[esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("chargeOperationVariablesOptimum", ip=2024)
timeseries_stor_charge = timeseries_stor_charge["values"]
timeseries_conv = esM.componentModelingDict[esM.componentNames["BHKW"]].getOptimalValues("operationVariablesOptimum", ip=2024)
timeseries_conv = timeseries_conv["values"]
timeseries_stor_soc = esM.componentModelingDict[esM.componentNames["Lithium-Ionen-Batterie"]].getOptimalValues("stateOfChargeOperationVariablesOptimum", ip=2024)
timeseries_stor_soc = timeseries_stor_soc["values"]

srcSnkSummary = pd.read_csv(f"srcSinkSummary_weight_co2_{weight_co2}.csv", dtype=dtype_dict)
convSummary = pd.read_csv(f"convSummary_weight_co2_{weight_co2}.csv", dtype=dtype_dict)
storSummary = pd.read_csv(f"storSummary_weight_co2_{weight_co2}.csv", dtype=dtype_dict)
timeseries_srcsnk = pd.read_csv(f"srcSinkTS_weight_co2_{weight_co2}.csv")
timeseries_conv = pd.read_csv(f"convTS_weight_co2_{weight_co2}.csv")
timeseries_stor_charge = pd.read_csv(f"storChargeTS_weight_co2_{weight_co2}.csv")
timeseries_stor_discharge = pd.read_csv(f"storDischargeTS_weight_co2_{weight_co2}.csv")
timeseries_stor_soc = pd.read_csv(f"storSOCTS_weight_co2_{weight_co2}.csv")

# ! timeseries_srcsnk entweder aus Ecvl-Dateine (mit  a-Wert) oder direkt aus FINE (irgendwein Wert)

timeseries_srcsnk.set_index("Unnamed: 0", inplace=True)
timeseries_stor_discharge.set_index("Unnamed: 0", inplace=True)
timeseries_stor_charge.set_index("Unnamed: 0", inplace=True)
timeseries_conv.set_index("Unnamed: 0", inplace=True)
timeseries_stor_soc.set_index('Unnamed: 0', inplace=True)

timeseries_srcsnk.drop(columns=["Unnamed: 1"], inplace=True)
timeseries_stor_discharge.drop(columns=["Unnamed: 1"], inplace=True)
timeseries_stor_charge.drop(columns=["Unnamed: 1"], inplace=True)
timeseries_conv.drop(columns=["Unnamed: 1"], inplace=True)
timeseries_stor_soc.drop(columns=["Unnamed: 1"], inplace=True)

strompreis = df_strom_pr_em_2024['Strompreis'].values

#strompreis = df_strom_pr_em_2024_sensi['Szenario Strompreis schwankt'].values # für Sensi

stromemissionen = df_strom_pr_em_2024['CO₂-Emissionsfaktor des Strommix'].values
strompreis_winter = strompreis[864:1032]
strompreis_sommer = strompreis[5232:5400]
stromemissionen_winter = stromemissionen[864:1032]
stromemissionen_sommer = stromemissionen[5232:5400]
gasemissionen_wert = 200.8 / (0.4225 + 0.4113)
gasemissionen = np.full(168, gasemissionen_wert)

batterie_soc = timeseries_stor_soc.loc['Lithium-Ionen-Batterie', :].values.flatten()
wasserspeicher_soc = timeseries_stor_soc.loc["Heizungswasserspeicher", :].values.flatten()
kältespeicher_soc=timeseries_stor_soc.loc["Kältespeicher", :].values.flatten()
stromkauf = timeseries_srcsnk.loc['Stromkauf von Spotmarkt', :].values.flatten()
pv = timeseries_srcsnk.loc['PV', :].values.flatten()
bhkw = timeseries_conv.loc['BHKW', :].values.flatten()
akm = timeseries_conv.loc['Absorptionskältemaschine', :].values.flatten()
kkm = timeseries_conv.loc['Kompressionskältemaschine', :].values.flatten()
batterie_discharge = timeseries_stor_discharge.loc['Lithium-Ionen-Batterie', :].values.flatten()
batterie_charge = timeseries_stor_charge.loc['Lithium-Ionen-Batterie', :].values.flatten()
#strombedarf = timeseries_srcsnk.loc['Strombedarf', :].values.flatten()
strombedarf = pv+stromkauf+bhkw

wärmepumpe = timeseries_conv.loc['Wärmepumpe', :].values.flatten()
wasserspeicher_discharge = timeseries_stor_discharge.loc['Heizungswasserspeicher', :].values.flatten()
wasserspeicher_charge = timeseries_stor_charge.loc['Heizungswasserspeicher', :].values.flatten()
kältespeicher_discharge = timeseries_stor_discharge.loc['Kältespeicher', :].values.flatten()
kältespeicher_charge = timeseries_stor_charge.loc['Kältespeicher', :].values.flatten()
wärmebedarf = timeseries_srcsnk.loc['Heizungswasserbedarf', :].values.flatten()
kältebedarf = timeseries_srcsnk.loc['Kältebedarf', :].values.flatten()

# ------------------------------------------------------------

# 10.2 Zeitspannen Sommer- und Winterwoche

stromkauf_winter = stromkauf[864:1032] # 6. bis 12. Februar Mo bis So
pv_winter = pv[864:1032]
bhkw_winter = bhkw[864:1032]
akm_winter = akm[864:1032]
kkm_winter = kkm[864:1032]
batterie_discharge_winter = batterie_discharge[864:1032]
batterie_charge_winter = batterie_charge[864:1032]
strombedarf_winter = strombedarf[864:1032]
wärmepumpe_winter = wärmepumpe[864:1032]
wärmebedarf_winter = wärmebedarf[864:1032]
kältebedarf_winter = kältebedarf[864:1032]
wasserspeicher_charge_winter = wasserspeicher_charge[864:1032]
wasserspeicher_discharge_winter = wasserspeicher_discharge[864:1032]
kältespeicher_charge_winter = kältespeicher_charge[864:1032]
kältespeicher_discharge_winter = kältespeicher_discharge[864:1032]



stromkauf_sommer = stromkauf[5064:5232] # 31. Juli bis 6. August Mo bis So
pv_sommer = pv[5064:5232]
bhkw_sommer = bhkw[5064:5232]
akm_sommer = akm[5064:5232]
kkm_sommer = kkm[5064:5232]
batterie_discharge_sommer = batterie_discharge[5064:5232]
batterie_charge_sommer = batterie_charge[5064:5232]
strombedarf_sommer = strombedarf[5064:5232]
wärmepumpe_sommer = wärmepumpe[5064:5232]
wärmebedarf_sommer = wärmebedarf[5064:5232]
kältebedarf_sommer = kältebedarf[5064:5232]
wasserspeicher_charge_sommer = wasserspeicher_charge[5064:5232]
wasserspeicher_discharge_sommer = wasserspeicher_discharge[5064:5232]
kältespeicher_charge_sommer = kältespeicher_charge[5064:5232]
kältespeicher_discharge_sommer = kältespeicher_discharge[5064:5232]

# ------------------------------------------------------------

# 10.3 Winterwoche Strom weight co2


def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

start_date = dt.datetime(2024, 2, 6)
dates_winter = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_winter))]
xticks_positions = range(0, len(dates_winter), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_winter[::24]]

fig1, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

stacked_data = pd.DataFrame({
    'Stromkauf': stromkauf_winter,
    'PV': pv_winter,
    'BHKW': bhkw_winter,
    'LIB (Entladung)': batterie_discharge_winter,
})

ax1.stackplot(range(len(stacked_data)), stacked_data.T,
              labels=stacked_data.columns, colors=[blau_stromkauf, gelb_pv, weinrot_bhkw, hellblau_lib])
ax1.plot(strombedarf_winter, label='Strombedarf', color=(8/255, 8/255, 8/255), linewidth=2)
ax1.plot(batterie_charge_winter, label='LIB (Ladung)', color=hellblau_lib, linewidth=2)

ax1.set_xlim(-2, 168)
ax1.set_ylabel('Energie in kWh', fontsize=11)
ax1.set_ylim(0, max(stromkauf.max(), strombedarf.max())+100)
fig1.legend(loc='upper center', bbox_to_anchor=(0.56, 1.02), ncol=3, fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
ax1.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"Winterwoche_Strom_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax2.plot(strompreis_winter, label='Strompreis', color=(8/255, 8/255, 8/255), linewidth=2)
ax2.plot(stromemissionen_winter, label='Stromemissionen', color=color_strom, linestyle='-.', linewidth=2)
ax2.plot(gasemissionen, label='Gasmissionen', color=color_raumwaerme, linestyle='--', linewidth=2)

ax2.set_xlim(-2, 168)
ax2.set_ylabel('Preis in €/kWh /\nEmissionen in g/$kWh_{el}$', fontsize=11)
ax2.set_ylim(min(strompreis_sommer.min(), strompreis_winter.min()), stromemissionen.max())
fig2.legend(loc='upper center', bbox_to_anchor=(0.56, 1.02), ncol=2, fontsize=11)
ax2.tick_params(axis='both', labelsize=11)
ax2.set_xticks(xticks_positions)
ax2.set_xticklabels(xticks_labels, rotation=45)
ax2.grid(axis='x', color='gray', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"Winterwoche_Strom_weight_co2_{weight_co2}_2.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 10.4 Sommerwoche Strom weight co2

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

start_date = dt.datetime(2024,7,31)
dates_sommer = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_sommer))]
xticks_positions = range(0, len(dates_sommer), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_sommer[::24]]

fig1, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

stacked_data = pd.DataFrame({
    'Stromkauf': stromkauf_sommer,
    'PV': pv_sommer,
    'BHKW': bhkw_sommer,
    'LIB (Entladung)': batterie_discharge_sommer,
})

ax1.stackplot(range(len(stacked_data)), stacked_data.T,
              labels=stacked_data.columns, colors=[blau_stromkauf, gelb_pv, weinrot_bhkw, hellblau_lib])
ax1.plot(strombedarf_sommer, label='Strombedarf', color=(8/255, 8/255, 8/255), linewidth=2)
ax1.plot(batterie_charge_sommer, label='LIB (Ladung)', color=hellblau_lib, linewidth=2)

ax1.set_xlim(-2, 168)
ax1.set_ylabel('Energie in kWh', fontsize=11)
ax1.set_ylim(0, max(stromkauf.max(), strombedarf.max())+100)
fig1.legend(loc='upper center', bbox_to_anchor=(0.56, 1.02), ncol=3, fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
ax1.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"Sommerwoche_Strom_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")

fig2, ax2 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax2.plot(strompreis_sommer, label='Strompreis', color=(8/255, 8/255, 8/255), linewidth=2)
ax2.plot(stromemissionen_sommer, label='Stromemissionen', color=color_strom, linestyle='-.', linewidth=2)
ax2.plot(gasemissionen, label='Gasemissionen', color=color_raumwaerme, linestyle='--', linewidth=2)

ax2.set_xlim(-2, 168)
ax2.set_ylabel('Preis in €/kWh /\nEmissionen in g/$kWh_{el}$', fontsize=11)
ax2.set_ylim(min(strompreis_sommer.min(), strompreis_winter.min())-10, stromemissionen.max())
fig2.legend(loc='upper center', bbox_to_anchor=(0.56, 1.02), ncol=2, fontsize=11)
ax2.tick_params(axis='both', labelsize=11)
ax2.set_xticks(xticks_positions)
ax2.set_xticklabels(xticks_labels, rotation=45)
ax2.grid(axis='x', color='gray', alpha=0.7)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(f"Sommerwoche_Strom_weight_co2_{weight_co2}_2.pdf", bbox_inches="tight")
plt.show()

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

start_date = dt.datetime(2024, 2, 6)
dates_winter = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_winter))]
xticks_positions = range(0, len(dates_winter), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_winter[::24]]

# ------------------------------------------------------------

# 10.5 Ladecharakterisierung Speichertechnologien

fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax1.fill_between(range(len(batterie_charge_winter)), batterie_charge_winter, label='Batterieaufladung', color=hellblau_lib)
ax1.set_xlim(-2, 168)
ax1.set_ylabel('Batterieaufladung in kWh', fontsize=11)
ax1.set_ylim(0, max(batterie_charge_sommer.max(), batterie_charge_winter.max()) + 100)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

ax2 = ax1.twinx()
ax2.plot(strompreis_winter, label='Strompreis', color=(8/255, 8/255, 8/255), linewidth=2)
ax2.set_ylim(min(strompreis_sommer.min(), strompreis_winter.min())-10, max(strompreis_sommer.max(), strompreis_winter.max()) + 50)
ax2.set_ylabel('Strompreis in €/kWh', fontsize=11)
ax2.set_xlim(-2, 168)
ax2.legend(loc='upper right', fontsize=11)
ax2.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig(f"Winterwoche_Batterie_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()

### Strom/Kälte

fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax1.fill_between(range(len(kältespeicher_charge_winter)), kältespeicher_charge_winter, label='Ladung Kältespeicher', color=hellblau_lib)
ax1.set_xlim(-2, 168)
ax1.set_ylabel('Ladung Kältespeicher in kWh', fontsize=11)
ax1.set_ylim(0, max(kältespeicher_charge_sommer.max(), kältespeicher_charge_winter.max()) + 100)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

ax2 = ax1.twinx()
ax2.plot(strompreis_winter, label='Strompreis', color=(8/255, 8/255, 8/255), linewidth=2)
ax2.set_ylim(min(strompreis_sommer.min(), strompreis_winter.min())-10, max(strompreis_sommer.max(), strompreis_winter.max()) + 50)
ax2.set_ylabel('Strompreis in €/kWh', fontsize=11)
ax2.set_xlim(-2, 168)
ax2.legend(loc='upper right', fontsize=11)
ax2.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig(f"Winterwoche_Kältespeicher_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()

### Strom/Kälte

fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax1.fill_between(range(len(wasserspeicher_charge_winter)), wasserspeicher_charge_winter, label='Ladung Wärmespeicher', color=hellblau_lib)
ax1.set_xlim(-2, 168)
ax1.set_ylabel('Ladung Wärmespeicher in kWh', fontsize=11)
ax1.set_ylim(0, max(wasserspeicher_charge_sommer.max(), wasserspeicher_charge_winter.max()) + 100)
ax1.grid(False)
ax1.legend(loc='upper left', fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

ax2 = ax1.twinx()
ax2.plot(strompreis_winter, label='Strompreis', color=(8/255, 8/255, 8/255), linewidth=2)
ax2.set_ylim(min(strompreis_sommer.min(), strompreis_winter.min())-10, max(strompreis_sommer.max(), strompreis_winter.max()) + 50)
ax2.set_ylabel('Strompreis in €/kWh', fontsize=11)
ax2.set_xlim(-2, 168)
ax2.legend(loc='upper right', fontsize=11)
ax2.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig(f"Winterwoche_Kältespeicher_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------

# 10.6 Winterwoche Raumwärme weight co2

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

start_date = dt.datetime(2024, 2, 6)
dates_winter = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_winter))]
xticks_positions = range(0, len(dates_winter), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_winter[::24]]

fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

stacked_data_winter = pd.DataFrame({
    'BHKW': bhkw_winter,
    "Wärmepumpe": wärmepumpe_winter,
    'Wasserspeicher (Entladung)': wasserspeicher_discharge_winter,
    'Wasserspeicher (Ladung)': wasserspeicher_charge_winter,
})

ax1.stackplot(range(len(stacked_data_winter)), stacked_data_winter.T,
              labels=stacked_data_winter.columns, colors=[weinrot_bhkw, orange_wp, grün_tes, color_kälte])
ax1.plot(wärmebedarf_winter, label='Wärmebedarf', color='black', linewidth=1.5)

ax1.set_xlim(-2, 168)
ax1.set_ylabel('Energie in kWh', fontsize=11)
ax1.set_ylim(0, max(wärmebedarf.max(), wasserspeicher_discharge_winter.max())+100)
fig.legend(loc='upper center', bbox_to_anchor=(0.56, 1.1), ncol=2, fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
ax1.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

plt.tight_layout()
plt.savefig(f"Winterwoche_Raumwärme_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 10.7 Sommerwoche Raumwärme weight co2

fig2, ax2 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax2.plot(wasserspeicher_charge_winter, label='Wasserspeicher (Ladung)', color=grün_tes, linewidth=1.5)

ax2.set_xlim(-2, 168)
ax2.set_ylabel('Energie in kWh', fontsize=11)
ax2.set_ylim(0, max(wärmebedarf.max(), wasserspeicher_discharge_winter.max())+100)
ax2.tick_params(axis='both', labelsize=11)
ax2.set_xticks(xticks_positions)
ax2.set_xticklabels(xticks_labels, rotation=45)
ax2.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

fig2.legend(loc='upper center', bbox_to_anchor=(0.56, 1.05), ncol=1, fontsize=11)

plt.tight_layout()
plt.savefig(f"Winterwoche_Raumwärme_weight_co2_{weight_co2}_2.pdf", bbox_inches="tight")
plt.show()

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

start_date = dt.datetime(2024,7, 31)
dates_sommer = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_winter))]
xticks_positions = range(0, len(dates_sommer), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_sommer[::24]]

fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

stacked_data_sommer = pd.DataFrame({
    'BHKW': bhkw_sommer,
    "Wärmepumpe": wärmepumpe_sommer,
    'Wasserspeicher (Entladung)': wasserspeicher_discharge_sommer,
    'Wasserspeicher (Ladung)': wasserspeicher_charge_sommer,
})

ax1.stackplot(range(len(stacked_data_sommer)), stacked_data_sommer.T,
              labels=stacked_data_sommer.columns, colors=[weinrot_bhkw, orange_wp, grün_tes, color_kälte])
ax1.plot(wärmebedarf_sommer, label='Wärmebedarf', color='black', linewidth=1.5)

ax1.set_xlim(-2, 168)
ax1.set_ylabel('Energie in kWh', fontsize=11)
ax1.set_ylim(0, max(wärmebedarf.max(), wasserspeicher_discharge_sommer.max())+100)
fig.legend(loc='upper center', bbox_to_anchor=(0.56, 1.1), ncol=2, fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
ax1.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

plt.tight_layout()
plt.savefig(f"Sommerwoche_Raumwärme_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")
plt.show()

fig2, ax2 = plt.subplots(figsize=(16/2.54, 12/2.54))

ax2.plot(wasserspeicher_charge_winter, label='Wasserspeicher (Ladung)', color=grün_tes, linewidth=1.5)

ax2.set_xlim(-2, 168)
ax2.set_ylabel('Energie in kWh', fontsize=11)
ax2.set_ylim(0, max(wärmebedarf.max(), wasserspeicher_discharge_sommer.max())+100)
ax2.tick_params(axis='both', labelsize=11)
ax2.set_xticks(xticks_positions)
ax2.set_xticklabels(xticks_labels, rotation=45)
ax2.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

fig2.legend(loc='upper center', bbox_to_anchor=(0.56, 1.05), ncol=1, fontsize=11)

plt.tight_layout()
plt.savefig(f"Sommerwoche_Raumwärme_weight_co2_{weight_co2}_2.pdf", bbox_inches="tight")
plt.show()

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

daily_mean_soc = np.mean(wasserspeicher_soc.reshape(-1, 24), axis=1)

dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(daily_mean_soc))]

# ------------------------------------------------------------

# 10.8 Winterwoche Kälte weight co2


start_date = dt.datetime(2024, 2, 6)
dates_winter = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_winter))]
xticks_positions = range(0, len(dates_winter), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_winter[::24]]


fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

stacked_data_winter = pd.DataFrame({
    'AKM': akm_winter,
    "KKM": kkm_winter,
    'Kältespeicher (Entladung)': kältespeicher_discharge_winter,
    'Kältespeicher (Ladung)': kältespeicher_charge_winter,
})

ax1.stackplot(range(len(stacked_data_winter)), stacked_data_winter.T,
              labels=stacked_data_winter.columns, colors=[hellgruen_AKM, pink_KKM, grün_tes, color_kälte])
ax1.plot(kältebedarf_winter, label='Wärmebedarf', color='black', linewidth=1.5)

ax1.set_xlim(-2, 168)
ax1.set_ylabel('Energie in kWh', fontsize=11)
ax1.set_ylim(0, max(wärmebedarf.max(), kältespeicher_discharge_winter.max())+100)
fig.legend(loc='upper center', bbox_to_anchor=(0.56, 1.1), ncol=2, fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
ax1.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

plt.tight_layout()
plt.savefig(f"Winterwoche_Kälte_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")
plt.show()

# Sommerfall

start_date = dt.datetime(2024,7, 31)
dates_sommer = [start_date + dt.timedelta(hours=i) for i in range(len(stromkauf_winter))]
xticks_positions = range(0, len(dates_sommer), 24)
xticks_labels = [date.strftime('%d.%m. (%a)') for date in dates_sommer[::24]]

fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

stacked_data_sommer = pd.DataFrame({
    'AKM': akm_sommer,
    "KKM": kkm_sommer,
    'Kältespeicher (Entladung)': kältespeicher_discharge_sommer,
    'Kältespeicher (Ladung)': kältespeicher_charge_sommer,
})

ax1.stackplot(range(len(stacked_data_sommer)), stacked_data_sommer.T,
              labels=stacked_data_sommer.columns, colors=[hellgruen_AKM, pink_KKM, grün_tes, color_kälte])
ax1.plot(kältebedarf_sommer, label='Wärmebedarf', color='black', linewidth=1.5)

ax1.set_xlim(-2, 168)
ax1.set_ylabel('Energie in kWh', fontsize=11)
ax1.set_ylim(0, max(kältebedarf.max(), kältespeicher_discharge_sommer.max())+100)
fig.legend(loc='upper center', bbox_to_anchor=(0.56, 1.1), ncol=2, fontsize=11)
ax1.tick_params(axis='both', labelsize=11)
ax1.set_xticks(xticks_positions)
ax1.set_xticklabels(xticks_labels, rotation=45)
ax1.grid(axis='x', color='gray', alpha=0.7)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))

plt.tight_layout()
plt.savefig(f"Sommerwoche_Kälte_weight_co2_{weight_co2}_1.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
# 10.9 Speicherfüllstände

# Füllstand des Heizungsspeichers
Jahressumme_wärme =wasserspeicher_soc.sum()

plt.figure(figsize=(15, 5))
integral_kwh = np.trapz(daily_mean_soc, dx=1)

plt.title('Füllstand des Heizungsspeichers', fontsize=11)

plt.ylabel('Füllstand in kWh$_{th}$', fontsize=11)
plt.ylim(0, daily_mean_soc.max()+1000)
plt.tick_params(axis='both', labelsize=11)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.xlim(dates[0], dates[-1])
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.fill_between(dates, daily_mean_soc, color=grün_tes, alpha=0.3)
plt.annotate(f'gespeichert imm Jahresverlauf ≈ {Jahressumme_wärme/1000:,.0f} MWh$_{{th}}$',
             xy=(0.65, 0.9), xycoords='axes fraction', fontsize=12)
plt.tight_layout()
plt.savefig(f"Wasserspeicher_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

daily_mean_soc = np.mean(kältespeicher_soc.reshape(-1, 24), axis=1)
Jahressumme_kälte=kältespeicher_soc.sum()
dates = [dt.datetime(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(daily_mean_soc))]

# Füllstand des Kältespeichers

plt.figure(figsize=(15, 5))
plt.plot(dates, daily_mean_soc, color=grün_tes)

plt.title('Füllstand des Kältespeichers', fontsize=11)

plt.ylabel('Füllstand in kWh$_{th}$', fontsize=11)
plt.ylim(0, daily_mean_soc.max()+1000)
plt.tick_params(axis='both', labelsize=11)

plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.fill_between(dates, daily_mean_soc, color=grün_tes, alpha=0.3)

plt.xlim(dates[0], dates[-1])
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.annotate(f'gespeichert imm Jahresverlauf ≈ {Jahressumme_kälte/1000:,.0f} MWh$_{{th}}$',
             xy=(0.65, 0.9), xycoords='axes fraction', fontsize=12)
plt.tight_layout()
plt.savefig(f"kältespeicher_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.show()

# ------------------------------------------------------------
