# Auswertung der Szenarienanalyse


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
from adjustText import adjust_text

cwd = Path.cwd()
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
grau_ks= (150 / 255, 150 / 255, 150 / 255)
lila_gas = (118 / 255, 0 / 255, 118 / 255)
weinrot_bhkw = (118 / 255, 0 / 255, 84 / 255)
hellgruen_AKM = ((50/255, 200/255, 125/255))
pink_KKM =((204/255, 0/255, 153/255))


color_raumwaerme = (176 / 255, 0 / 255, 70 / 255)  # RGB (176, 0, 70)
color_raumwaerme_light = (243 / 255, 217 / 255, 227 / 255)
color_strom = (0 / 255, 83 / 255, 116 / 255)
color_strom_light = (217 / 255, 229 / 255, 234 / 255)
# -------------------------------------------------------------
# 2. Daten einlesen

# 2.1 Liste der Unterordner
ordner = [
    "Erhöhung_Strompreise",
    "Verringerung_Strompreise",
    "Neubau_Kältemaschinen",
    "Verdopplung_Heizungsspeicher",
    "Verdopplung_Kältespeicher",
    "Basisszenario",
]

abkuerzungen = [
    "Strompreis steigt",  # Strompreis steigt
    "Strompreis\nsinkt",   # Strompreis sinkt
    "Neubau\nKältemaschinen",  # Strompreis schwankt stärker
    "Verdopplung\nHeizungsspeicher",  # Wärmepumpe
    "Verdopplung\nKältespeicher",  # Lithium-Ionen-Batterie
    "Basisszenario"
]
# -------------------------------------------------------------

# 2.2 TAC und THGE für Paretofront einlesen

tac_values_zsmfssg = []
co2_values_zsmfssg = []
labels_zsmfssg = []
colors_zsmfssg = []


def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")


for ordner_name, abkuerzung in zip(ordner, abkuerzungen):
    dateipfad = cwd / ordner_name / "Paretofront.csv"

    # CSV-Datei einlesen
    if dateipfad.exists():
        df = pd.read_csv(dateipfad)

        # Alle TAC und CO2 Emissionen auslesen und umrechnen
        tac_values_zsmfssg = df["TAC [€]"].values / 1000  # TAC in Tausend Euro
        co2_values_zsmfssg = df["CO2 Emissions [g]"].values / 1e6  # CO2 in Tonnen

        # Farbe festlegen
        if abkuerzung == "Basisszenario":
            colors_zsmfssg = color_strom
            linewidth = 2  # Dickere Linie für das Basisszenario
        else:
            colors_zsmfssg = color_raumwaerme
            linewidth = 1.5  # Standard-Linienstärke

        # Linie und Punkte plotten
        plt.plot(tac_values_zsmfssg, co2_values_zsmfssg, color=colors_zsmfssg, linestyle='-',
                 linewidth=linewidth)  # marker='o'

        # Nur den ersten Punkt beschriften
        if abkuerzung in ["Strompreis steigt"]:
            # Beschrifte den letzten Punkt
            plt.text(tac_values_zsmfssg[-1] - 1000, co2_values_zsmfssg[-1] + 4000, abkuerzung, fontsize=11, ha='left',
                     va='bottom')
        elif abkuerzung in ["Strompreis\nsinkt"]:
            plt.text(tac_values_zsmfssg[0], co2_values_zsmfssg[0] - 5000, abkuerzung, fontsize=11, ha='left', va='bottom')
        elif abkuerzung in ["Neubau\nKältemaschinen"]:
            plt.text(tac_values_zsmfssg[0]+600, co2_values_zsmfssg[0] - 6000, abkuerzung, fontsize=11, ha='left',
                     va='bottom')
        #elif abkuerzung in ["Verdopplung\nHeizungsspeicher"]:
        #    plt.text(tac_values_zsmfssg[0]+550, co2_values_zsmfssg[0] - 1250, abkuerzung, fontsize=11, ha="left",
        #             va="bottom")
        #elif abkuerzung in ["Basisszenario"]:
        #    plt.text(tac_values_zsmfssg[0]+550, co2_values_zsmfssg[0] - 5500, abkuerzung, fontsize=11, ha="left",
        #             va="bottom")
        #elif abkuerzung in ["Verdopplung\nKältespeicher"]:
        #    plt.text(tac_values_zsmfssg[0]+1200, co2_values_zsmfssg[0] - 3250, abkuerzung, fontsize=11, ha="left",
        #             va="bottom")
        #else:
        #    # Beschrifte den ersten Punkt
        #    plt.text(tac_values_zsmfssg[0] + 15, co2_values_zsmfssg[0] - 80, abkuerzung, fontsize=11, ha='left',
        #             va='bottom')

# Dummy-Punkte für die Legende hinzufügen
plt.plot([], [], color=color_strom, label='Basisszenario')
plt.plot([], [], color=color_raumwaerme, label='Sensitivitätsanalyse')

# Legende anzeigen
plt.legend(fontsize=11, loc="upper right")

# Achsenbeschriftungen
plt.xlabel('TAC in Tausend €', fontsize=12)
plt.ylabel('CO$_2$-Emissionen in t', fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.gca().xaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.tight_layout()
plt.savefig(f"Abb5_17_Zsmfssg_Sensitivitätsanalysen.svg")
plt.show()
# -------------------------------------------------------------

# 2.3 Ausgelegte Leistungen einlesen


# Create empty dictionaries to store capacities
cap_pv = []
cap_bhkw = []
cap_akm = []
cap_kkm = []
cap_hp = []
cap_tes = []
cap_ks = []
cap_lib = []
weights_list = []

# Define weights for CO2
weights = np.linspace(0, 0.4, 11)

# Loop over ordners and weights
for ordner_name in ordner:
    for weight_co2_zsmfssg in weights:
        # Load the CSV files for srcSnkSummary, convSummary, and storSummary
        src_sink_summary_zsmfssg = pd.read_csv(
            cwd / ordner_name / f"srcSinkSummary_weight_co2_{weight_co2_zsmfssg}.csv")
        conv_summary_zsmfssg = pd.read_csv(cwd / ordner_name / f"convSummary_weight_co2_{weight_co2_zsmfssg}.csv")
        stor_summary_zsmfssg = pd.read_csv(cwd / ordner_name / f"storSummary_weight_co2_{weight_co2_zsmfssg}.csv")

        # Extract capacities for each technology
        cap_pv.append(src_sink_summary_zsmfssg.loc[(src_sink_summary_zsmfssg['Component'] == 'PV') &
                                                   (src_sink_summary_zsmfssg['Property'] == 'capacity') &
                                                   (src_sink_summary_zsmfssg[
                                                        'Unit'] == '[kW_el]'), 'FabrikAachen'].sum())

        cap_bhkw.append(conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'BHKW') &
                                                 (conv_summary_zsmfssg['Property'] == 'capacity') &
                                                 (conv_summary_zsmfssg['Unit'] == '[kW_el]'), 'FabrikAachen'].sum())

        cap_hp.append(conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'Wärmepumpe') &
                                               (conv_summary_zsmfssg['Property'] == 'capacity') &
                                               (conv_summary_zsmfssg['Unit'] == '[kW_th]'), 'FabrikAachen'].sum())

        cap_kkm.append(conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'Kompressionskältemaschine') &
                                           (conv_summary_zsmfssg['Property'] == 'capacity') &
                                           (conv_summary_zsmfssg['Unit'] == '[kW_th_c]'), 'FabrikAachen'].sum())

        cap_akm.append(conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'Absorptionskältemaschine') &
                                           (conv_summary_zsmfssg['Property'] == 'capacity') &
                                           (conv_summary_zsmfssg['Unit'] == '[kW_th_c]'), 'FabrikAachen'].sum())

        cap_tes.append(stor_summary_zsmfssg.loc[(stor_summary_zsmfssg['Component'] == 'Heizungswasserspeicher') &
                                                (stor_summary_zsmfssg['Property'] == 'capacity') &
                                                (stor_summary_zsmfssg['Unit'] == '[kW_th*h]'), 'FabrikAachen'].sum())

        cap_ks.append(stor_summary_zsmfssg.loc[(stor_summary_zsmfssg['Component'] == 'Kältespeicher') &
                                                (stor_summary_zsmfssg['Property'] == 'capacity') &
                                                (stor_summary_zsmfssg['Unit'] == '[kW_th_c*h]'), 'FabrikAachen'].sum())

        cap_lib.append(stor_summary_zsmfssg.loc[(stor_summary_zsmfssg['Component'] == 'Lithium-Ionen-Batterie') &
                                                (stor_summary_zsmfssg['Property'] == 'capacity') &
                                                (stor_summary_zsmfssg['Unit'] == '[kW_el*h]'), 'FabrikAachen'].sum())

        weights_list.append(weight_co2_zsmfssg)

# Convert capacities lists to MW (except for TES and LIB which are in MWh)
cap_pv = np.array(cap_pv) / 1000
cap_bhkw = np.array(cap_bhkw) / 1000
cap_kkm = np.array(cap_kkm) / 1000
cap_akm = np.array(cap_akm) / 1000
cap_hp = np.array(cap_hp) / 1000
cap_tes = np.array(cap_tes) / 1000
cap_ks = np.array(cap_ks) / 1000
cap_lib = np.array(cap_lib) / 1000
weights_list = np.array(weights_list)

# Create a DataFrame for plotting
data = pd.DataFrame({
    'PV-Anlage [MW$_{el}$]': cap_pv,
    'BHKW [MW$_{el}$]': cap_bhkw,
    'AKM [MW$_{th}$]': cap_akm,
    'KKM [MW$_{th}$]': cap_kkm,
    'Wärmepumpe [MW$_{th}$]': cap_hp,
    'Heizungswasserspeicher [MWh$_{th}$]': cap_tes,
    'Kältespeicher [MWh$_{th}$]': cap_ks,
    'LIB [MWh$_{el}$]': cap_lib
})

# Melt the DataFrame to get it in long format for seaborn
data_melted = data.melt(var_name='Technology', value_name='Capacity')

data_corrmatrix = pd.DataFrame({
    'PV-Anlage [MW$_{el}$]': cap_pv,
    'BHKW [MW$_{el}$]': cap_bhkw,
    'AKM [MW$_{th}$]': cap_akm,
    'KKM [MW$_{th}$]': cap_kkm,
    'Wärmepumpe [MW$_{th}$]': cap_hp,
    'Kältespeicher [MWh$_{th}$]': cap_ks,
    'Heizungswasserspeicher [MWh$_{th}$]': cap_tes,
    'LIB [MWh$_{el}$]': cap_lib,
    "CO$_2$-Gewichtungsfaktor": weights_list
})

weights = np.linspace(0, 0.4, 11)
weight_co2_zsmfssg = weights[5]

cap_pv_list = []
cap_bhkw_list = []
cap_hp_list = []
cap_kkm_list = []
cap_akm_list = []
cap_tes_list = []
cap_ks_list = []
cap_lib_list = []

for ordner_name in ordner:
    # Lade die CSV-Dateien für srcSnkSummary, convSummary und storSummary
    src_sink_summary_zsmfssg = pd.read_csv(cwd / ordner_name / f"srcSinkSummary_weight_co2_{weight_co2_zsmfssg}.csv")
    conv_summary_zsmfssg = pd.read_csv(cwd / ordner_name / f"convSummary_weight_co2_{weight_co2_zsmfssg}.csv")
    stor_summary_zsmfssg = pd.read_csv(cwd / ordner_name / f"storSummary_weight_co2_{weight_co2_zsmfssg}.csv")

    cap_pv = src_sink_summary_zsmfssg.loc[(src_sink_summary_zsmfssg['Component'] == 'PV') &
                                          (src_sink_summary_zsmfssg['Property'] == 'capacity') &
                                          (src_sink_summary_zsmfssg['Unit'] == '[kW_el]'), 'FabrikAachen'].sum()

    cap_bhkw = conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'BHKW') &
                                        (conv_summary_zsmfssg['Property'] == 'capacity') &
                                        (conv_summary_zsmfssg['Unit'] == '[kW_el]'), 'FabrikAachen'].sum()

    cap_kkm = conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'Kompressionskältemaschine') &
                                        (conv_summary_zsmfssg['Property'] == 'capacity') &
                                        (conv_summary_zsmfssg['Unit'] == '[kW_th_c]'), 'FabrikAachen'].sum()

    cap_akm = conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'Absorptionskältemaschine') &
                                        (conv_summary_zsmfssg['Property'] == 'capacity') &
                                        (conv_summary_zsmfssg['Unit'] == '[kW_th_c]'), 'FabrikAachen'].sum()

    cap_hp = conv_summary_zsmfssg.loc[(conv_summary_zsmfssg['Component'] == 'Wärmepumpe') &
                                      (conv_summary_zsmfssg['Property'] == 'capacity') &
                                      (conv_summary_zsmfssg['Unit'] == '[kW_th]'), 'FabrikAachen'].sum()

    cap_tes = stor_summary_zsmfssg.loc[(stor_summary_zsmfssg['Component'] == 'Heizungswasserspeicher') &
                                       (stor_summary_zsmfssg['Property'] == 'capacity') &
                                       (stor_summary_zsmfssg['Unit'] == '[kW_th*h]'), 'FabrikAachen'].sum()

    cap_ks = stor_summary_zsmfssg.loc[(stor_summary_zsmfssg['Component'] == 'Kältespeicher') &
                                       (stor_summary_zsmfssg['Property'] == 'capacity') &
                                       (stor_summary_zsmfssg['Unit'] == '[kW_th_c*h]'), 'FabrikAachen'].sum()

    cap_lib = stor_summary_zsmfssg.loc[(stor_summary_zsmfssg['Component'] == 'Lithium-Ionen-Batterie') &
                                       (stor_summary_zsmfssg['Property'] == 'capacity') &
                                       (stor_summary_zsmfssg['Unit'] == '[kW_el*h]'), 'FabrikAachen'].sum()
    print(f"{ordner_name}: KKM={cap_kkm}, AKM={cap_akm}")

    # Capacities zur Liste hinzufügen
    cap_pv_list.append(cap_pv / 1000)  # Umrechnung in MW
    cap_bhkw_list.append(cap_bhkw / 1000)  # Umrechnung in MW
    cap_hp_list.append(cap_hp / 1000)  # Umrechnung in MW
    cap_kkm_list.append(cap_kkm / 1000)  # Umrechnung in MW
    cap_akm_list.append(cap_akm / 1000)  # Umrechnung in MW
    cap_tes_list.append(cap_tes / 1000)  # Umrechnung in MWh
    cap_ks_list.append(cap_ks / 1000)  # Umrechnung in MWh
    cap_lib_list.append(cap_lib / 1000)  # Umrechnung in MWh
    print(f"KKM={cap_kkm_list}, AKM={cap_akm_list}")

# -------------------------------------------------------------

# 2.4 ausgelegte Leistungen plotten


x = np.arange(len(ordner))  # Anzahl der Ordner
bar_width = 0.35

fig, ax1 = plt.subplots(figsize=(16 / 2.54, 12 / 2.54))

# Erstelle die Balken für die Technologien mit der Einheit kW auf der ersten Achse
ax1.bar(x - bar_width / 2, cap_pv_list, width=bar_width, color=gelb_pv, label='PV-Anlage [MW$_{el}$]')
ax1.bar(x - bar_width / 2, cap_bhkw_list, width=bar_width, bottom=cap_pv_list, color=weinrot_bhkw,
        label='BHKW [MW$_{el}$]')
ax1.bar(x - bar_width / 2, cap_hp_list, width=bar_width, bottom=np.array(cap_pv_list) + np.array(cap_bhkw_list),
        color=orange_wp, label='Wärmepumpe [MW$_{th}$]')
ax1.bar(x - bar_width / 2, cap_kkm_list, width=bar_width, bottom=np.array(cap_pv_list) + np.array(cap_bhkw_list) + np.array(cap_hp_list),
        color=pink_KKM, label='KKM [MW$_{th}$]')
ax1.bar(x - bar_width / 2, cap_akm_list, width=bar_width, bottom=np.array(cap_pv_list) + np.array(cap_bhkw_list) + np.array(cap_hp_list) + np.array(cap_kkm_list),
        color=hellgruen_AKM, label='AKM [MW$_{th}$]')

# Beschriftung der ersten y-Achse und Titel
ax1.set_ylabel('Leistung in MW', fontsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(abkuerzungen, rotation=45, ha='right', fontsize=10)
ax1.tick_params(axis='y', labelsize=11)

# Erstelle die zweite y-Achse für TES und LIB
ax2 = ax1.twinx()
ax2.bar(x + bar_width / 2, cap_tes_list, width=bar_width, color=grün_tes, label='Heizungswasserspeicher [MWh$_{th}$]')
ax2.bar(x + bar_width / 2, cap_ks_list, width=bar_width, bottom=np.array(cap_tes_list), color=color_kälte, label='Kältespeicher [MWh$_{th}$]')
ax2.bar(x + bar_width / 2, cap_lib_list, width=bar_width, bottom=np.array(cap_ks_list)+np.array(cap_tes_list), color=hellblau_lib,
        label='LIB [MWh$_{el}$]')

# Beschriftung der zweiten y-Achse
ax2.set_ylabel('Kapazität in MWh', fontsize=11)
ax2.tick_params(axis='y', labelsize=11)

# Erstelle Legenden für beide Achsen
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=10)

# Diagramm anzeigen
plt.tight_layout(rect=[0, 0, 1, 0.9])
ax1.set_ylim(0, (max(cap_pv_list)+max(cap_bhkw_list)+max(cap_hp_list)))
ax2.set_ylim(0, (max(cap_tes_list)+max(cap_lib_list)+max(cap_ks_list)) * 1.1)
plt.savefig(f"Abb5_18_Zsmfssg_Kapazitäten_weight_CO2_{weight_co2_zsmfssg}.pdf", bbox_inches='tight')
plt.show()

palette_ax1 = [gelb_pv, weinrot_bhkw,hellgruen_AKM ,pink_KKM, orange_wp]  # Colors for PV, BHKW, Wärmepumpe
palette_ax2 = [grün_tes,color_kälte, hellblau_lib]            # Colors for Heizungswasserspeicher, LIB

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")


fig, ax1 = plt.subplots(figsize=(16/2.54, 12/2.54))

sns.violinplot(x='Technology', y='Capacity', data=data_melted[data_melted['Technology'].isin(['PV-Anlage [MW$_{el}$]', 'BHKW [MW$_{el}$]','AKM [MW$_{th}$]','KKM [MW$_{th}$]', 'Wärmepumpe [MW$_{th}$]'])],
                bw_adjust=0.5, ax=ax1, palette=palette_ax1, cut=0, inner="quart")
ax1.set_ylabel('Leistung in MW', fontsize=11)
ax1.set_xlabel('')
ax1.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])  # Manually setting the ticks for the first axis
ax1.set_xticklabels(['PV-Anlage\n[MW$_{el}$]', 'BHKW\n[MW$_{el}$]', 'AKM\n[MW$_{th}$]','KKM\n[MW$_{th}$]','Wärmepumpe\n[MW$_{th}$]', 'Heizungswassersp.\n[MWh$_{th}$]', 'Kältespeicher\n[MWh$_{th}$]', 'LIB\n[MWh$_{el}$]'], rotation=45, fontsize=11)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'.replace('.', ',')))

ax2 = ax1.twinx()

sns.violinplot(x='Technology', y='Capacity', data=data_melted[data_melted['Technology'].isin(['Heizungswasserspeicher [MWh$_{th}$]','Kältespeicher [MWh$_{th}$]', 'LIB [MWh$_{el}$]'])],
               bw_adjust=0.5, ax=ax2, palette=palette_ax2, cut=0, inner="quart")
ax2.set_ylabel('Kapazität in MWh', fontsize=11)

#plt.title('Kapazitätsverteilungen aus allen Szenarien für Technologien', fontsize=13)
plt.tight_layout()
plt.ylim(0, data_melted["Capacity"].max() * 1.1)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.savefig(f"Abb5_23_Violinplots_Capacities.pdf")
plt.show()

# -------------------------------------------------------------

# 2.5 Einlesen/plotten Strom- und Erdgaskauf

weight_co2_zsmfssg = 0.2

# Listen für Strom- und Gasoperationen
strom_list = []
gas_list = []

for ordner_name in ordner:
    # Lade die CSV-Datei für srcSnkSummary
    src_sink_summary_zsmfssg = pd.read_csv(cwd / ordner_name / f"srcSinkSummary_weight_co2_{weight_co2_zsmfssg}.csv")

    # Extrahiere den Stromkauf und den Erdgaskauf
    operation_strom = src_sink_summary_zsmfssg.loc[
                          (src_sink_summary_zsmfssg['Component'] == 'Stromkauf von Spotmarkt') &
                          (src_sink_summary_zsmfssg['Property'] == 'operation') &
                          (src_sink_summary_zsmfssg['Unit'] == '[kW_el*h/a]'), 'FabrikAachen'].sum() / 1e6

    operation_gas = src_sink_summary_zsmfssg.loc[
                        (src_sink_summary_zsmfssg['Component'] == 'Erdgaskauf') &
                        (src_sink_summary_zsmfssg['Property'] == 'operation') &
                        (src_sink_summary_zsmfssg['Unit'] == '[kW_CH4,LHV*h/a]'), 'FabrikAachen'].sum() / 1e6

    # Operationen zur Liste hinzufügen
    strom_list.append(operation_strom)
    gas_list.append(operation_gas)

# Erstelle das gestapelte Säulendiagramm
x = np.arange(len(ordner))  # Anzahl der Ordner


fig, ax = plt.subplots(figsize=(16 / 2.54, 12 / 2.54))

# Gestapelte Balken
bar1 = ax.bar(x, strom_list, width=0.5, color=blau_stromkauf, label='Strombezug [GWh$_{el}$]')
bar2 = ax.bar(x, gas_list, width=0.5, bottom=strom_list, color=lila_gas, label='Gasbezug [GWh$_{CH_4}$]')

# Achsenbeschriftungen
ax.set_ylabel('Jährlich eingekaufte Endenergie in GWh', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(abkuerzungen, rotation=45, ha='right', fontsize=11)
ax.tick_params(axis='y', labelsize=11)

# Legende oberhalb
fig.legend(
    [bar1, bar2],
    ['Strombezug [GWh$_{el}$]', 'Gasbezug [GWh$_{CH_4}$]'],
    loc='upper center',
    bbox_to_anchor=(0.5, 1),
    ncol=2,
    fontsize=10,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig(f"Abb5_19_Zsmfssg_Strom_Gas_Operation_weight_CO2_{weight_co2_zsmfssg}.pdf", bbox_inches='tight')
plt.show()
# -------------------------------------------------------------
