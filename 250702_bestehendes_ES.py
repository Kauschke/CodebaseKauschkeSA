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
from pathlib import Path
import os
import warnings
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from IPython.display import display
import datetime as dt
import matplotlib.dates as mdates


cwd = Path.cwd()

np.random.seed(
    42
)  # Sets a "seed" to produce the same random input data in each model run

stromverbr = pd.read_pickle("stromverbrauch.pkl")
heizungswasserverb = pd.read_pickle("heizungswasser.pkl")
strompreis_cost = pd.read_pickle("strompreis_cost.pkl")
strompreis_revenue = pd.read_pickle("strompreis_revenue.pkl")
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


print((kältebed <= 0).sum())

df_strom_pr_em_2024 = pd.read_pickle("StromPreisEmissionen.pkl") # €/MWh
df_strom_pr_em_2024_sensi = pd.read_pickle("StromPreisEmissionenSensi.pkl")

#stromemissionen.mean()
#stromemissionen=0.143*stromemissionen
#stromemissionen.mean()

# Input parameters
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
numberOfTimeSteps, hoursPerTimeStep = 8784, 1 # Index geht von 0 bis 8783
costUnit, lengthUnit = "Euro", "m" # 1e3  wären 1000
startYear = 2024

#Wärme von BHKW ausgeben

# Code
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
        capacityMax = 5000,  # max. Netzanschlussleistung 5 MW
        investPerCapacity = 51.53  # €/kW
        #investPerCapacity = 214.59   # €/kW
    )
)

esM.add(
    fn.Source(
        esM=esM,
        name="Erdgaskauf",
        commodity="erdgas",
        hasCapacityVariable=True,
        commodityCostTimeSeries=gaspreis, # €/kWh
        #commodityCost=0.0336

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
        hasCapacityVariable=False
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
        operationRateMin=0.5,
        economicLifetime=20-10, # remainingEconomicLifetime 10a # default 10
        investPerCapacity=0, # wird in den nächsten Jahren abgeschrieben, 800€/kW_el
        opexPerCapacity=800*0.02, # 16€/kW_el
        opexPerOperation=0.008, # €/kWh_el
        # yearlyFullLoadHoursMin?
    )
)



capacity_kw = 1990
efficiency = 1
hours_in_year = 8784

index = range(0, hours_in_year)
generation_series_wärme = pd.Series([capacity_kw * efficiency] * len(index), index=index)
differenz_wärme_erzeugung_bedarf = generation_series_wärme - heizungswasserverb
print(f"Stunden mit zu wenig BHKW-Erzeugung {(differenz_wärme_erzeugung_bedarf <= 0).sum()}")
print(f"Minimum der Differenz {differenz_wärme_erzeugung_bedarf.min()} --> für Dimensionierung Wasserspeicher ")
print(f"Welcher Teil des Wärmebedarfs kann nie aus BHKW direkt gedeckt werden? {(abs(differenz_wärme_erzeugung_bedarf[differenz_wärme_erzeugung_bedarf < 0]).sum()/heizungswasserverb.sum())}")
print(f"Welcher Wärmebedarf kann nie aus BHKW direkt gedeckt werden? {abs(differenz_wärme_erzeugung_bedarf[differenz_wärme_erzeugung_bedarf < 0]).sum()}")
# der Überschuss ist Prozessdmapf+Kälte
plt.plot(differenz_wärme_erzeugung_bedarf, linewidth=0.5)
plt.ylabel("kW")
plt.xlabel("Stunden in 2023")
#plt.title("BHKW-Erzeugung - Wärmebedarf")
plt.show()

differenz_wärme_erzeugung_bedarf.max()

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
        capacityMax=420,           # thermische Nennleistung
        interestRate=0.05,
        economicLifetime=20 - 10,  # remainingEconomicLifetime 10a # default 10
        investPerCapacity=0,
        opexPerCapacity=300 * 0.01,  # €/kW_th
        opexPerOperation=0.015,  # €/kW_th
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
        #capacityMax=300,           # thermische Nennleistung
        capacityMax=300,  # thermische Nennleistung
        interestRate=0.05,
        economicLifetime=20 - 10,  # remainingEconomicLifetime 10a # default 10
        investPerCapacity=0,  # wird in den nächsten Jahren abgeschrieben, 800€/kW_el
        opexPerCapacity=100 * 0.02,  # €/kW_el
        #opexPerOperation=0.25,  # €/kWh_el
        opexPerOperation=0.015,  # €/kWh_el

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
        selfDischarge=0.0026,
        hasCapacityVariable=True,
        interestRate=0.05,
        economicLifetime=30,
        investPerCapacity=10, # €/kWh
        opexPerCapacity=10*0.015, # €/kWh
        hasIsBuiltBinaryVariable=True,
        bigM=375,
        capacityMax=375,
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
        yearlyLimit= 9999 * 1000 * 1000, # erste Zahl in t denken und dann umrechnen in g
        #yearlyLimit=5004938271.6049385,
        commodityCost=45/(1000*1000), # 30 €/t CO2-Preis
    )
)

esM.add(
    fn.Sink(
        esM=esM,
        name="Notkühler BHKW",
        commodity="heizungswasser",
        hasCapacityVariable=False,
        commodityCost=0.05, # 5 € pro kW_th Strafe, weil BHKW auf 50 % P_min geregelt ist
    )
)

# Annahme komisch, dass BHKW 24/7 durchläuft

capacity_kw = 1990
efficiency = 1      # da elektrischer und thermischer Wirkungsgrad gleich sind
hours_in_year = 8784

index = range(0, hours_in_year)
generation_series = pd.Series([capacity_kw * efficiency] * len(index), index=index)
stromüberschuss = generation_series + pv*2000 - stromverbr
plt.figure(figsize=(16/2.54, 12/2.54))
plt.plot(stromüberschuss)
plt.xlabel("Stunden in 2024", fontsize=11)
plt.ylabel("Leistung in kW", fontsize=11)


#Einlesen der Daten:
cd=r'C:\Users\andri\OneDrive\Studium\Master TU BS\SoSe_2025\Studienarbeit\Daten'
path = os.path.join(cd, "250423_Energiedaten_2024_komplett.xlsx")
path_bhkw = os.path.join(cd, "250423_Energiedaten_307_2024.xlsx")
path_strom_pr_em = os.path.join(cd, "241231_StromPreisEmissionsfaktorAgora.csv")

# Suppress the warning about the default style
warnings.simplefilter(action='ignore', category=UserWarning)

df = pd.read_excel(path)
df.set_index('Datum', inplace=True)
df.index = pd.to_datetime(df.index, format='%d.%m.%Y  %H:%M')

# df_2023 = df[df.index.year == 2023] # diesen Filter nicht mehr, da in der Excel nur 2023er Daten drin sind, aber der "Bis-Zeitstempel" also ab 1.1.23 1 Uhr bis 1.1.24 0 Uhr
df_2024 = df

df_2024.reset_index(drop=True, inplace=True)
df_2024.index = df_2024.index + 0 # macht Index ab 0, ... + 1 macht Index ab 1

# Zahlenformat 1000,00 bereits richtig konvertiert in 1000.00

def process_series(ser):
    ser = ser.copy()

    if not isinstance(ser, pd.Series):
        raise TypeError('Die Eingabe muss eine pandas Series sein.')

    def is_outlier(value, lower_bound, upper_bound):
        return value < lower_bound or value > upper_bound

    # Anzahl der NaN-Werte vor dem Setzen von Extremwerten zu NaN
    num_nans_before = ser.isna().sum()
    print(f'Anzahl der NaN-Werte vor dem Setzen der Extremwerte auf NaN: {num_nans_before}')

    # Extremwerte (Maximum und Minimum)
    max_value = ser.max(skipna=True)
    min_value = ser.min(skipna=True)
    print(f'Maximum: {max_value}')
    print(f'Minimum: {min_value}')

    # Median als Referenz (Nullwerte rausrechnen, da BHKW-Zeitreihen viele 0-Werte enthalten)
    ser_non_zero = ser[ser != 0]
    median_value = ser_non_zero.median(skipna=True)
    print(f'Median: {median_value}')

    # IQR-Methode zur Ausreißererkennung
    Q1 = ser.quantile(0.25)
    Q3 = ser.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Überprüfe, ob Extremwerte Ausreißer sind
    max_is_outlier = is_outlier(max_value, lower_bound, upper_bound)
    min_is_outlier = is_outlier(min_value, lower_bound, upper_bound)

    if max_is_outlier:
        response = input(f'Das Maximum ({max_value}) ist ein Ausreißer. Soll es ersetzt werden? (ja/nein): ')
        if response.lower() == 'ja':
            ser[ser == max_value] = np.nan

    if min_is_outlier:
        response = input(f'Das Minimum ({min_value}) ist ein Ausreißer. Soll es ersetzt werden? (ja/nein): ')
        if response.lower() == 'ja':
            ser[ser == min_value] = np.nan

    # Anzahl der NaN-Werte nach dem Setzen von Extremwerten zu NaN
    num_nans_after = ser.isna().sum()
    print(f'Anzahl der NaN-Werte nach dem Setzen der Extremwerte auf NaN: {num_nans_after}')

    # Interpoliere alle NaN-Werte auf einmal
    ser_final_interpolated = ser.interpolate(method='linear')

    return ser_final_interpolated

stromeinspeisung_real = process_series(df_2024["Stromzähler W5 20kV-Ausspeisung Kaubendenstr. (Geb. 809) - ∅ - kW"])
print(stromeinspeisung_real.sum())
print((pv*2000).sum())
print(stromeinspeisung_real.max())

plt.figure(figsize=(16/2.54, 12/2.54))
plt.plot(stromüberschuss, label="Theoretisch möglicher Stromüberschuss")
plt.plot(pv*2000, label="PV-Erzeugung")
plt.plot(stromeinspeisung_real, label="reale Stromeinspeisung in 2024")
plt.xlabel("Stunden in 2024", fontsize=11)
plt.ylabel("Leistung in kW", fontsize=11)
plt.legend(fontsize=11)

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

esM.componentNames

timeSeriesAggregation=False
solver="gurobi"

esM.verboseLogLevel = 3  # maximale Ausführlichkeit

model=esM.optimize(timeSeriesAggregation=timeSeriesAggregation, solver=solver)
print("Optimierte Jahre:", esM.investmentPeriods)



srcSnkSummary = esM.getOptimizationSummary("SourceSinkModel",  ip=2024, outputLevel=1)
display(esM.getOptimizationSummary("SourceSinkModel", ip=2024, outputLevel=2))
tac_srcsnk = srcSnkSummary.xs('TAC', level='Property').sum().sum()
co2_srcsnk = srcSnkSummary.xs(('operation', '[gCO2/h*h/a]'), level=('Property', 'Unit')).loc['CO2 in Atmosphäre'].sum()
display(tac_srcsnk)
display(co2_srcsnk)

convSummary = esM.getOptimizationSummary("ConversionModel", ip=2024, outputLevel=1)
tac_conv = convSummary.xs('TAC', level='Property').sum().sum()
display(esM.getOptimizationSummary("ConversionModel", ip=2024, outputLevel=2))
display(tac_conv)

storSummary = esM.getOptimizationSummary("StorageModel", ip=2024, outputLevel=1)
tac_stor = storSummary.xs('TAC', level='Property').sum().sum()
display(esM.getOptimizationSummary("StorageModel", ip=2024, outputLevel=2))
display(tac_stor)

tac_total =tac_stor+  tac_conv + tac_srcsnk
display(tac_total)

# Farben der TU Braunschweig nach Schema RGB

color_kälte = (118/255, 0/255, 118/255)  # lila
color_strom = (0/255, 83/255, 116/255)  # blau
color_raumwaerme = (176/255, 0/255, 70/255)  # rot
color_raumwaerme_light = (215/255, 127/255, 162/255) # hellrot
color_strom_light = (140/255, 198/255, 221/255) # hellblau
gelb_pv = (255/255, 205/255, 0/255)
rot_raumwaerme = color_raumwaerme
blau_stromkauf = (0/255, 128/255, 180/255)
dunkelblau_strom = color_strom
grün_tes = (0/255, 113/255, 86/255)
lila_gas = (118/255, 0/255, 118/255)
weinrot_bhkw = (118/255, 0/255, 84/255)
hellgruen_AKM = ((50 / 255, 200 / 255, 125 / 255))
pink_KKM =((204/255, 0/255, 153/255))
orange_nk=((250/255, 110/255, 0/255))


cap_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "capacity", "[kW_el]")].sum()
cap_pv = srcSnkSummary.loc[("PV", "capacity", "[kW_el]")].sum()
cap_bhkw = convSummary.loc[("BHKW", "capacity", "[kW_el]")].sum()
cap_tes_h = storSummary.loc[("Heizungswasserspeicher", "capacity", "[kW_th*h]")].sum()
cap_KKM = convSummary.loc[("Kompressionskältemaschine", "capacity", "[kW_th_c]")].sum()
cap_AKM = convSummary.loc[("Absorptionskältemaschine", "capacity", "[kW_th_c]")].sum()



categories = ['Netzanschluss-\nleistung\n[kW$_{el}$]', 'PV-Anlage\n[kW$_{el}$],\ngegeben', 'BHKW\n[kW$_{el}$]*', 'Heizungswassersp.\n[kWh$_{th}$]',  'KKM\n[kWh$_{th}$]', 'AKM\n[kWh$_{th}$]']
capacities = [cap_strom, cap_pv, cap_bhkw,cap_tes_h, cap_KKM, cap_AKM]

y_buffer=max(capacities)*1.25
plt.figure(figsize=(16/2.54, 16/2.54))
bars = plt.bar(categories, capacities, color=[color_strom, gelb_pv, weinrot_bhkw,grün_tes, pink_KKM, hellgruen_AKM])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, f'{yval:.0f}', ha='center', va='bottom', fontsize=11)



#plt.xlabel('Technologie', fontsize=12)
plt.ylabel('Leistung bzw. Kapazität [kW bzw. kWh]', fontsize=11)
plt.ylim(0, y_buffer)
plt.title('Gegebene und optimal ausgebaute Leistungen/Kapazitäten', fontsize=11)
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.figtext(0.1, 0.1, '*Die thermische und elektrische Leistung des BHKW sind circa gleich groß,\nda der elektrische und der thermische Wirkungsgrad annähernd gleich gut sind.', ha='left', va='top', fontsize=9)

plt.tight_layout(rect=[0, 0.1, 1, 1])  # unten mehr Platz
#plt.savefig(f"Kapazitäten_weight_co2_{weight_co2}.pdf", bbox_inches='tight')
plt.show()

tac_pv = srcSnkSummary.loc[("PV", "TAC", "[Euro/a]")].sum() # v. a. CAPEX
tac_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "TAC", "[Euro/a]")].sum() # OPEX
tac_gas = srcSnkSummary.loc[("Erdgaskauf", "TAC", "[Euro/a]")].sum() # OPEX
tac_bhkw = convSummary.loc[("BHKW", "TAC", "[Euro/a]")].sum() # OPEX
tac_AKM= convSummary.loc[("Absorptionskältemaschine", "TAC", "[Euro/a]")].sum() # OPEX
tac_KKM= convSummary.loc[("Kompressionskältemaschine", "TAC", "[Euro/a]")].sum() # OPEX
tac_tes_h = storSummary.loc[("Heizungswasserspeicher", "TAC", "[Euro/a]")].sum() # v. a. CAPEX
tac_co2 = srcSnkSummary.loc[("CO2 in Atmosphäre", "commodCosts", "[Euro/a]")].sum() # nur OPEX
tac_NK = srcSnkSummary.loc[("Notkühler BHKW", "commodCosts", "[Euro/a]")].sum() # nur OPEX

opex_pv = srcSnkSummary.loc[("PV", "opexCap", "[Euro/a]")].sum() + srcSnkSummary.loc[("PV", "commodCosts", "[Euro/a]")].sum()

opex_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "commodCosts", "[Euro/a]")].sum() - srcSnkSummary.loc[("Stromkauf von Spotmarkt", "commodRevenues", "[Euro/a]")].sum()
opex_gas = srcSnkSummary.loc[("Erdgaskauf", "commodCosts", "[Euro/a]")].sum()
opex_bhkw = convSummary.loc[("BHKW", "opexCap", "[Euro/a]")].sum() + convSummary.loc[("BHKW", "opexOp", "[Euro/a]")].sum()
opex_AKM = convSummary.loc[("Absorptionskältemaschine", "opexCap", "[Euro/a]")].sum() + convSummary.loc[("Absorptionskältemaschine", "opexOp", "[Euro/a]")].sum()
opex_KKM = convSummary.loc[("Kompressionskältemaschine", "opexCap", "[Euro/a]")].sum() + convSummary.loc[("Kompressionskältemaschine", "opexOp", "[Euro/a]")].sum()
opex_tes_h = storSummary.loc[("Heizungswasserspeicher", "opexCap", "[Euro/a]")].sum()
opex_co2 = srcSnkSummary.loc[("CO2 in Atmosphäre", "commodCosts", "[Euro/a]")].sum()
opex_NK = srcSnkSummary.loc[("Notkühler BHKW", "commodCosts", "[Euro/a]")].sum()


capex_pv = srcSnkSummary.loc[("PV", "capexCap", "[Euro/a]")].sum()
capex_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "capexCap", "[Euro/a]")].sum()
capex_gas = 0
capex_bhkw = convSummary.loc[("BHKW", "capexCap", "[Euro/a]")].sum()
capex_AKM = convSummary.loc[("Absorptionskältemaschine", "capexCap", "[Euro/a]")].sum()
capex_tes_h = storSummary.loc[("Heizungswasserspeicher", "capexCap", "[Euro/a]")].sum()
capex_KKM = convSummary.loc[("Kompressionskältemaschine", "capexCap", "[Euro/a]")].sum()
capex_co2 = 0
capex_NK = 0

print(f'Die Betriebskosten AKM sind: {opex_AKM}')
print(f'Die TAC AKM sind: {tac_AKM}')
print(f'Die Investitionskosten AKM sind: {capex_AKM}')

# Werte durch 1000 teilen
opex_pv, opex_strom, opex_gas, opex_bhkw, opex_AKM, opex_KKM,opex_tes_h, opex_co2, opex_NK = [x / 1000 for x in [opex_pv, opex_strom, opex_gas, opex_bhkw, opex_AKM, opex_KKM,opex_tes_h,  opex_co2, opex_NK]]
capex_pv, capex_strom, capex_gas, capex_bhkw, capex_AKM, capex_KKM,capex_tes_h, capex_co2, capex_NK = [x / 1000 for x in [capex_pv, capex_strom, capex_gas, capex_bhkw,capex_AKM, capex_KKM,capex_tes_h, capex_co2, capex_NK]]
tac_pv, tac_strom, tac_gas, tac_bhkw,tac_AKM, tac_KKM,tac_tes_h,  tac_co2, tac_NK = [x / 1000 for x in [tac_pv, tac_strom, tac_gas, tac_bhkw,tac_AKM, tac_KKM,tac_tes_h, tac_co2, tac_NK]]

costs = { # €/Jahr
    'Technologie': ['PV-Anlage', 'Stromkauf', 'Erdgaskauf', 'BHKW', 'AKM', 'KKM','Heizungs-\nwassersp.', "$CO_2$-Preis", 'Notkühler'],
    'OPEX': [opex_pv, opex_strom, opex_gas, opex_bhkw, opex_AKM, opex_KKM,opex_tes_h, opex_co2, opex_NK],
    'CAPEX': [capex_pv, capex_strom, capex_gas, capex_bhkw,capex_AKM ,capex_KKM,capex_tes_h, capex_co2, capex_NK],
    'TAC': [tac_pv, tac_strom, tac_gas, tac_bhkw,tac_AKM, tac_KKM,tac_tes_h, tac_co2, tac_NK]
}

df_costs = pd.DataFrame(costs)

print(f'Der leistungspreis beträgt {capex_strom}')

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

print(df_costs[df_costs['Technologie'].isin(['AKM', 'KKM'])])

df_plot = df_costs.copy()
df_plot = df_plot.set_index("Technologie")[["OPEX", "CAPEX"]]
x = np.arange(len(df_plot))
width = 0.25  # Breite der Balken

fig, ax = plt.subplots(figsize=(16/2.54, 12/2.54))
df_costs.set_index('Technologie')[['OPEX', 'CAPEX']].plot(kind='bar', stacked=True, ax=ax, color= [color_strom, color_raumwaerme])

for i, j in enumerate(df_costs['TAC']):
    tac_formatted = f'{j:,.1f}'.replace(',', 'TEMP').replace('.', ',').replace('TEMP', '.')
    ax.text(i, j + 15, tac_formatted, ha='center', fontsize=11) # i is the x-coordinate where the text will be placed, corresponding to the position of the bar. j + 50 is the y-coordinate where the text will be placed, slightly above the top of the bar. f'{j}' is the text that will be displayed, which is the total value from the TAC column.

#ax.bar(x + width, df_plot["TAC"], width=width, label='TAC', color='yellow')

#plt.xlabel('Technologie', fontsize=12)
ax.set_xlabel("")
plt.ylabel('Kosten in Tausend €', fontsize=11)
#plt.title('OPEX und CAPEX pro Modellkomponente', fontsize=13)
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.legend(fontsize=11)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))  # Y-Achse mit Tausendertrennpunkten
plt.savefig(f"Abb5_22_TAC_StatusQuo.pdf", bbox_inches="tight")
plt.show()

operation_notkühler = srcSnkSummary.loc[("Notkühler BHKW", "operation", "[kW_th*h/a]")].sum()

print(f'Der Notkühler hat eine Jahresleistung von {operation_notkühler} kWh')
operation_strombedarf = srcSnkSummary.loc[("Strombedarf", "operation", "[kW_el*h/a]")].sum()
operation_heizungswasserbedarf = srcSnkSummary.loc[("Heizungswasserbedarf", "operation", "[kW_th*h/a]")].sum()
operation_kältebedarf=srcSnkSummary.loc[("Kältebedarf", "operation", "[kW_th_c*h/a]")].sum()
operation_pv = srcSnkSummary.loc[("PV", "operation", "[kW_el*h/a]")].sum()
operation_strom = srcSnkSummary.loc[("Stromkauf von Spotmarkt", "operation", "[kW_el*h/a]")].sum()
operation_gas = srcSnkSummary.loc[("Erdgaskauf", "operation", "[kW_CH4,LHV*h/a]")].sum()
operation_bhkw = convSummary.loc[("BHKW", "operation", "[kW_el*h/a]")].sum()
operation_AKM = convSummary.loc[("Absorptionskältemaschine", "operation", "[kW_th_c*h/a]")].sum()
operation_KKM = convSummary.loc[("Kompressionskältemaschine", "operation", "[kW_th_c*h/a]")].sum()
#operation_heizungswasserspeicher = (storSummary.loc[("Heizungswasserspeicher", "operationCharge", "[kW_th*h/a]")].sum() + storSummary.loc[("Heizungswasserspeicher", "operationDischarge", "[kW_th*h/a]")].sum())/2 # Mittelwert
operation_heizungswasserspeicher = storSummary.loc[("Heizungswasserspeicher", "operationDischarge", "[kW_th*h/a]")].sum() # Mittelwert
jahressummen = [x / 1000000 for x in [operation_pv, operation_strom, operation_gas, operation_bhkw,operation_AKM ,operation_KKM, operation_heizungswasserspeicher, operation_notkühler]]
jahresbedarfe = [x / 1000000 for x in [operation_strombedarf, operation_heizungswasserbedarf, operation_kältebedarf]]

categories2 = ['PV-Anlage [GWh$_{el}$]', "Strombezug [GWh$_{el}$]", "Gasbezug [GWh$_{CH_4}$]", 'BHKW [GWh$_{el}$]','AKM [GWh$_{th_c}$]', 'KKM [GWh$_{th_c}$]', 'Heizungswassersp. [GWh$_{th}$]', 'Notkühler BHKW [GWh$_{th}$]']
categories3 = ['Strom [GWh$_{el}$]', 'Raumwärme [GWh$_{th}$]', 'Kälte [GWh$_{th_c}$]']

fig = plt.figure(figsize=(26/2.54, 24/2.54))
plt.subplots_adjust(left=0.3)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

# Erstes Diagramm
ax1 = plt.subplot(gs[0])
bars1 = ax1.barh(categories2, jahressummen, color=[gelb_pv, blau_stromkauf, lila_gas, weinrot_bhkw, hellgruen_AKM, pink_KKM, grün_tes, orange_nk])
for bar in bars1:
    xval = bar.get_width()
    ax1.text(xval + 0.5, bar.get_y() + bar.get_height() / 2, f'{xval:.2f}', ha='left', va='center', fontsize=11)
ax1.set_xlabel('Jahresmenge', fontsize=12)
ax1.set_title('Energiemengen', fontsize=13)
ax1.set_yticks(ax1.get_yticks())
ax1.set_yticklabels(categories2, rotation=0, fontsize=11)

# Zweites Diagramm
ax2 = plt.subplot(gs[1])
bars2 = ax2.barh(categories3, jahresbedarfe, color=[color_strom, color_raumwaerme, color_kälte])
for bar in bars2:
    xval = bar.get_width()
    ax2.text(xval + 0.1, bar.get_y() + bar.get_height() / 2, f'{xval:.2f}', ha='left', va='center', fontsize=11)
ax2.set_xlabel('Jahresbedarf', fontsize=12)
ax2.set_title('Strom- und Raumwärmebedarf', fontsize=13)
ax2.set_yticks(ax2.get_yticks())
ax2.set_yticklabels(categories3, rotation=0, fontsize=11)

max_x = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
ax1.set_xlim(0, max_x+1)
ax2.set_xlim(0, max_x+1)

#plt.figtext(0.22, 0.05, '*Die thermische und elektrische Leistung des BHKW sind circa gleich groß, da der elektrische und der thermische\nWirkungsgrad annähernd gleich gut sind.', ha='left', va='top', fontsize=9)


plt.tight_layout()
#plt.savefig(f"operation_weight_co2_{weight_co2}.pdf", bbox_inches="tight")
plt.savefig(f"Übersicht Status Quo.pdf", bbox_inches="tight")
plt.show()

# Ausgabe der jährlichen TAC und THGE:

#tac_NK = srcSnkSummary.loc[("Notkühler BHKW", "commodCosts", "[Euro/a]")].sum() # nur OPEX

display(tac_total)
display(co2_srcsnk)

fig = plt.figure(figsize=(6, 4))
plt.subplots_adjust(left=0.3)
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
ax1 = plt.subplot(gs[0])
bars1 = ax1.barh(['THGE [t CO₂]', 'TAC [tausend €]'], [co2_srcsnk/(1000*1000), tac_total/1000], color=[gelb_pv, blau_stromkauf])
for bar in bars1:
    xval = bar.get_width()
    ax1.text(xval + 50, bar.get_y() + bar.get_height() / 2, f'{xval:.2f}', ha='left', va='center', fontsize=12)
ax1.set_xlabel('Wert', fontsize=12)
ax1.set_title('jährliche TAC und THGE', fontsize=12)
ax1.set_xlim(0,8000)
plt.tight_layout()
#plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.show()

### Berechnung der KPIs

# Anteil des Selbsterzeugten Stroms/ Autakiequote=verbrauchter Solarstrom/gesamter Stromverbrauch

selbsterzeugungsanteil=operation_pv/operation_strombedarf

print(f'Der Selbsterzeugungsanteil durch PV beträgt {selbsterzeugungsanteil:.2%}')

#PV-Nutzungsquote Eigenverbrauchsquote
einspeisung_pv = srcSnkSummary.loc[("Überschussstrom in Netz", "operation", "[kW_el*h/a]")].sum()
print(f"PV-Einspeisung ins Netz: {einspeisung_pv / 1e6:.2f} GWh")

eigenverbrauchsquote=(operation_pv-einspeisung_pv)/operation_pv
print(f'Die PV-Nutzungsquote  PV beträgt {eigenverbrauchsquote:.2%}')


kpi_labels = ['Selbsterzeugungsanteil', 'PV-Eigenverbrauchsquote']
kpi_values = [selbsterzeugungsanteil, eigenverbrauchsquote]

plt.figure(figsize=(6, 4))
bars = plt.bar(kpi_labels, kpi_values, color=['green', 'blue'])
plt.ylabel('Anteil [1 = 100 %]')
plt.ylim(0, 1)
plt.title('PV-Kennzahlen')

# Prozentwerte direkt auf die Balken schreiben
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.1%}',
             ha='center', va='bottom')

plt.tight_layout()
plt.show()



### Piecharts Energiebedarfe


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

# Strommengen in kWh
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

# Aufteilung TAC


labels = ['PV-Anlage', 'Stromkauf', 'Erdgaskauf', 'BHKW', 'AKM', 'KKM','Heizungs-\nwassersp.', "$CO_2$-Preis"]
sizes = [tac_pv, tac_strom, tac_gas, tac_bhkw, tac_AKM, tac_KKM, tac_tes_h, tac_co2]
colors = [gelb_pv, color_strom, lila_gas, weinrot_bhkw, hellgruen_AKM, pink_KKM, grün_tes, color_raumwaerme]
explode = [0, 0, 0, 0.05, 0.1, 0.1, 0.15, 0.1]

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

'''
# Verteilung CO2 Emissionen
co2_gesamt = co2_srcsnk
co2_gas = operation_gas * co2_faktor_gas
co2_strom = co2_gesamt - co2_gas
co2_gesamt_gerundet = int(co2_gesamt/1000000)


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
### Strombezug über den Jahresverlauf


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

## Test

operationValues = esM.componentModelingDict[esM.componentNames["Stromkauf von Spotmarkt"]].getOptimalValues("operationVariablesOptimum", ip=2024)

# Nur den Spotmarkt extrahieren:
#operation_spotmarkt = operationValues["Stromkauf von Spotmarkt"]
# Jetzt Maximum bestimmen:
#max_spotmarkt = operation_spotmarkt.max()
#print(f"Maximale Leistung Stromkauf von Spotmarkt: {max_spotmarkt:.2f} kW")

### Heatmaps:

custom_cmap_rot = LinearSegmentedColormap.from_list("weiß_zu_raumwaerme", [color_strom, gelb_pv])
#custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme", [(124/255, 205/255, 239/255), (190/255, 30/255, 70/255)])
custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme", [color_strom, color_strom, gelb_pv])



fig1, ax1 = fn.plotOperationColorMap(
    esM,
    "CO2 in Atmosphäre",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="THGE [g]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot,
    save=True,
    fileName='CO2_in_Atmosphäre.jpeg'
    )

cbar = fig1.get_axes()[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))

fig2, ax2 = fn.plotOperationColorMap(
    esM,
    "Notkühler BHKW",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Leistung Notkühler [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot,
    )
cbar = fig2.get_axes()[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))

fig3, ax3 = fn.plotOperationColorMap(
    esM,
    "PV",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="PV-Erzeugung[kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

cbar = fig3.get_axes()[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))

fig4, ax4  = fn.plotOperationColorMap(
    esM,
    "Stromkauf von Spotmarkt",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Stromkauf von Spotmarkt [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )
cbar = fig4.get_axes()[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))

fig5, ax5 = fn.plotOperationColorMap(
    esM,
    "Strombedarf",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Strombedarf [kW]",
    orientation="horizontal",
    fontsize=12.,
    cmap=custom_cmap_rot
    )

cbar = fig5.get_axes()[-1]
cbar.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))

fig, ax = fn.plotOperationColorMap(
    esM,
    "Erdgaskauf",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Erdgasbezug[kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

fig, ax = fn.plotOperationColorMap(
    esM,
    "BHKW",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="BHKW-Betrieb [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

fig, ax = fn.plotOperationColorMap(
    esM,
    "Kompressionskältemaschine",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Leistung der KKM [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

fig, ax = fn.plotOperationColorMap(
    esM,
    "Absorptionskältemaschine",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Leistung der AKM [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

fig, ax = fn.plotOperationColorMap(
    esM,
    "Heizungswasserbedarf",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Raumwärmebedarf [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )


fig, ax = fn.plotOperationColorMap(
    esM,
    "Kältebedarf",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Kältebedarf [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

fig, ax = fn.plotOperationColorMap(
    esM,
    "Überschussstrom in Netz",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="Stromverkauf [kW]",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

fig, ax = fn.plotOperationColorMap(
    esM,
    "Heizungswasserspeicher",
    "FabrikAachen",
    nbPeriods=366,
    nbTimeStepsPerPeriod=24,
    ip=2024,
    figsize=(16/2.54, 12/2.54),
    xlabel="Tag des Jahres",
    ylabel="Stunde des Tages",
    zlabel="SOC des Heizungswassersp. [kWh_th]",
    variableName="stateOfChargeOperationVariablesOptimum",
    orientation="horizontal",
    fontsize=12,
    cmap=custom_cmap_rot
    )

plt.show()


#-------------------------- Line Plots--------------------------------
