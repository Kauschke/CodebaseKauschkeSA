# Einlesen der Inputdaten


# 1. Grundlagen


# 1.1 Import der Bibilotheken


import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.ticker import FuncFormatter, StrMethodFormatter
import warnings
import matplotlib.ticker as ticker


#mit Komma und 2 Nachkommastellen
def comma_formatter(x, pos):
    return "{:,.2f}".format(x).replace(".", ",")
formatter = FuncFormatter(comma_formatter)

np.random.seed(42)
# -------------------------------------------------------------


# 1.2 standartisierte Farben:


color_strom = (0 / 255, 83 / 255, 116 / 255)  # RGB (0, 83, 116)
color_raumwaerme = (176 / 255, 0 / 255, 70 / 255)  # RGB (176, 0, 70)
#color_kälte = (100/255, 50/255, 200/255)  # RGB (176, 0, 70)
color_kälte = (124/255, 205/255, 230/255)  # RGB (176, 0, 70)

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
# -------------------------------------------------------------



# 2. Einlesen der Daten:

# 2.1 Grünenthal-Daten als Excel einlesen

cd=r'C:\Users\andri\OneDrive\Studium\Master TU BS\SoSe_2025\Studienarbeit\Daten\250524_Anpassung_Zählerdaten'
path = os.path.join(cd, "250524_Energiedaten_2024_komplett.xls")
path_bhkw = os.path.join(cd, "250524_Energiedaten_307_2024.xlsx")
path_strom_pr_em = os.path.join(cd, "241231_StromPreisEmissionsfaktorAgora.csv")

# Suppress the warning about the default style
warnings.simplefilter(action='ignore', category=UserWarning)

df = pd.read_excel(path)
df.set_index('Datum', inplace=True)
df.index = pd.to_datetime(df.index, format='%d.%m.%Y  %H:%M')

df_2024 = df
df_2024.reset_index(drop=True, inplace=True)
df_2024.index = df_2024.index + 0 # macht Index ab 0, ... + 1 macht Index ab 1

# Suppress the warning about the default style
warnings.simplefilter(action='ignore', category=UserWarning)

df_bhkw = pd.read_excel(path_bhkw)
df_bhkw.set_index('Datum', inplace=True)
df_bhkw.index = pd.to_datetime(df_bhkw.index, format='%d.%m.%Y  %H:%M')

# df_bhkw_2023 = df_bhkw[df_bhkw.index.year == 2023] # diesen Filter nicht mehr, da in der Excel nur 2023er Daten drin sind, aber der "Bis-Zeitstempel" also ab 1.1.23 1 Uhr bis 1.1.24 0 Uhr
df_bhkw_2024 = df_bhkw
df_bhkw_2024.reset_index(drop=True, inplace=True)
df_bhkw_2024.index = df_bhkw_2024.index + 0 # macht Index ab 0, ... + 1 macht Index ab 1

# Zahlenformat 1000,00 bereits richtig konvertiert in 1000.00

columns_df_2024 = set(df_2024.columns)
columns_df_bhkw_2024 = set(df_bhkw_2024.columns)

unique_columns_df_2024 = columns_df_2024 - columns_df_bhkw_2024

unique_columns_df_bhkw_2024 = columns_df_bhkw_2024 - columns_df_2024

unique_columns_df_2024_list = list(unique_columns_df_2024)
unique_columns_df_bhkw_2024_list = list(unique_columns_df_bhkw_2024)

print("Spalten, die nur in df_2024 vorkommen:", unique_columns_df_2024_list)
print("Spalten, die nur in df_bhkw_2024 vorkommen:", unique_columns_df_bhkw_2024_list)
# -------------------------------------------------------------



# 2.2  csv mit Strompreisen und -emissionen einlesen (Quelle: Agora Energiewende)

df_strom_pr_em = pd.read_csv(path_strom_pr_em)
print(df.columns.tolist())
print(df_strom_pr_em.columns)
df_strom_pr_em.set_index('Datetime', inplace=True)
df_strom_pr_em.index = pd.to_datetime(df_strom_pr_em.index, format='%Y-%m-%dT%H:%M:%S')
df_strom_pr_em_2024 = df_strom_pr_em[df_strom_pr_em.index.year == 2024]


# Umgang mit NaN values
full_date_range = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='h')
missing_dates = full_date_range.difference(df_strom_pr_em_2024.index)
missing_dates_count = len(missing_dates)
print(missing_dates_count)
print(missing_dates)

df_strom_pr_em_2024 = df_strom_pr_em_2024.reindex(full_date_range)
df_strom_pr_em_2024 = df_strom_pr_em_2024.interpolate(method='linear')
df_strom_pr_em_2024.reset_index(drop=True, inplace=True)
df_strom_pr_em_2024.index = df_strom_pr_em_2024.index
df_strom_pr_em_2024.to_pickle("StromPreisEmissionen.pkl")
# -------------------------------------------------------------

# 3 Energiebedarfe separat abspeichern

# 3.1 Umverteilung der 1 MW Zählerwerte auf vorherige 0er Einträge

def distribute_energy_series(series):
    series = series.copy()
    last_idx = None

    for idx in series.index:
        value = series.loc[idx]

        if value > 0:
            if last_idx is not None:
                steps = series.index[last_idx + 1: idx + 1]  # inkl. aktueller Impulszeit
                if len(steps) > 0:
                    energy_per_step = value / len(steps)
                    series.loc[steps] = energy_per_step
            # Setze aktuellen Index als Start für nächsten Block
            last_idx = idx

    return series
# -------------------------------------------------------------


# 3.2 Zuweisung der Spalten im Excel-Sheet zu den Energieträgern

strom = df_bhkw_2024["Stromzähler Geb. 307 BHKW Generator Einspeisung - ∅ - kW"] + df_2024["Stromzähler W5 20kV-Einspeisung Kaubendenstr. (Geb. 809) - ∅ - kW"]
heizungswasser = distribute_energy_series(df_bhkw_2024["Heizungswasser von BHKW307-Netz zu Geb. 201 301 407 - ∅ - kW"])+distribute_energy_series(df_bhkw_2024["Heizungswasser Geb. 307 Absorbtionskälte - ∅ - kW"])

# Der Prozessdampf für das eine Gebäude liegt bei 120-150 kW und wird konstant als 140 kW angenommen und vom Heißwasser abgezogen, um die Heizlast zu berechnen.
heisswasser = distribute_energy_series(df_bhkw_2024["Heißwassererzeugung Geb. 307 aus BHKW - ∅ - kW"]) - 140
#heisswasser = distribute_energy_series(df_bhkw_2024["Heißwassererzeugung Geb. 307 gesamt - ∅ - kW"]) - 490
heisswasser = heisswasser.clip(lower=0) # Alle Werte, die nach dem Abzug von 490 kW negativ wären, werden auf 0 gesetzt.
heizungswasser = heizungswasser + heisswasser

#Kälte=df_bhkw_2024["Kaltwassererzeugung Geb. 307 Absorptionskälte - ∅ - kW"]+eer_KKM*(df_bhkw_2024['Stromzähler Geb. 307 Kältemaschine 2 - ∅ - kW']+df_bhkw_2024['Stromzähler Geb. 307 Kältemaschine 3 - ∅ - kW']+df_bhkw_2024['Stromzähler Geb. 307 Kältemaschine 4 - ∅ - kW'])
Kälte=df_bhkw_2024["Kaltwasser Geb. 307 gesamt - ∅ - kW"]

# -------------------------------------------------------------

# 4. Datenaufbereitung (Ausreißer aus Datenreihe entfernen)


# 4.1 process_series für Extremwerte

# Um vor dem Setzen der Extremwerte als NaN eine Abfrage zu machen, ob diese Extremwerte Ausreißer sind, kann die Verwendung des Interquartilabstands (IQR) helfen.
# Werte, die außerhalb von 1,5 mal dem IQR über dem dritten Quartil oder unter dem ersten Quartil liegen, werden üblicherweise als Ausreißer betrachtet.

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

# -------------------------------------------------------------

# 4.2 Ausreißeruntersuchung, qualitativ

strom = process_series(strom)
Kälte=process_series(Kälte)
heizungswasser = process_series(heizungswasser)

filtered_indices = heizungswasser[heizungswasser > 3000].index
print(filtered_indices)

stromtest = df_bhkw_2024["Stromzähler Geb. 307 BHKW Generator Einspeisung - ∅ - kW"] + df_2024["Stromzähler W5 20kV-Einspeisung Kaubendenstr. (Geb. 809) - ∅ - kW"]
heizungswassertest_all = df_bhkw_2024["Heizungswasser von BHKW307-Netz zu Geb. 201 301 407 - ∅ - kW"] + df_bhkw_2024["Heizungswasser Geb. 307 Absorbtionskälte - ∅ - kW"] + df_bhkw_2024["Heizungswasser Geb. 307 Notkühler - ∅ - kW"]
heizungswassertest = df_bhkw_2024["Heizungswasser von BHKW307-Netz zu Geb. 201 301 407 - ∅ - kW"]
heisswassertest = df_bhkw_2024["Heißwassererzeugung Geb. 307 aus BHKW - ∅ - kW"]

heisswassertest = process_series(heisswassertest)
heizungswassertest_all = process_series(heizungswassertest_all)
heizungswassertest = process_series(heizungswassertest)
stromtest = process_series(stromtest)
idx=198

# Testplot
plt.figure(figsize=(16/2.54, 12/2.54))
plt.stackplot(stromtest.index, stromtest, heizungswassertest_all, heisswassertest, labels=['Strom', 'Heizungswasser, gesamt', 'Heißwasser'], colors=['blue', 'red', 'orange'])
# plt.plot(gastest.index, gastest, label='Gas', color='green', linewidth=2)
plt.legend(fontsize=11)
plt.xlim(idx - 198, idx + 198)
plt.xlabel("Stunden", fontsize=12)
plt.ylabel("kW", fontsize=12)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
plt.title("Ausreißeruntersuchung Heizungswasser", fontsize=13)
plt.show()
# -------------------------------------------------------------

# 4.3 process_outliers für Ausreißer

def process_outliers(ser):
    if not isinstance(ser, pd.Series):
        raise TypeError('Die Eingabe muss eine pandas Series sein.')

    def is_outlier(value, lower_bound, upper_bound):
        return value < lower_bound or value > upper_bound

    # IQR-Methode zur Ausreißererkennung
    Q1 = ser.quantile(0.25)
    Q3 = ser.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Median als Referenz
    median_value = ser.median(skipna=True)
    print(f'Median: {median_value}')

    # Überprüfe alle Daten auf Ausreißer und setze sie auf NaN
    outliers = ser.apply(lambda x: np.nan if is_outlier(x, lower_bound, upper_bound) else x)

    # Ausgabe der Ausreißer und ihrer Indizes
    outlier_values = ser[~ser.isin(outliers)]
    print(f'Ausreißer-Werte: {outlier_values.dropna().tolist()}')
    outlier_indices = outlier_values.index
    print(f'Ausreißer-Indizes: {outlier_indices.tolist()}')
    num_outliers = len(outlier_values.dropna())
    print(f'Anzahl der Ausreißer: {num_outliers}')

    response = input(f'Sollen die Ausreißer ersetzt werden? (ja/nein): ')
    if response.lower() == 'ja':
        # Anzahl der NaN-Werte nach dem Setzen der Ausreißer zu NaN
        num_nans_after = outliers.isna().sum()
        print(f'Anzahl der NaN-Werte nach dem Setzen aller Ausreißer auf NaN: {num_nans_after}')

        # Interpoliere alle NaN-Werte auf einmal
        ser_final_interpolated = outliers.interpolate(method='linear')
        return ser_final_interpolated

    return ser
# -------------------------------------------------------------

# 4.4 process_outliers für Zeitreihen mit vielen Nullwerten


def process_outliers_000(ser):
    if not isinstance(ser, pd.Series):
        raise TypeError('Die Eingabe muss eine pandas Series sein.')

    def is_outlier(value, lower_bound, upper_bound):
        return value < lower_bound or value > upper_bound

    # Filtern der Serie, um Nullwerte auszuschließen
    ser_non_zero = ser[ser != 0]

    # IQR-Methode zur Ausreißererkennung nur auf die Serie ohne 0-Werte anwenden
    Q1 = ser_non_zero.quantile(0.25)
    Q3 = ser_non_zero.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Median als Referenz
    median_value = ser_non_zero.median(skipna=True)
    print(f'Median: {median_value}')

    # Überprüfe alle Daten auf Ausreißer und setze sie auf NaN
    outliers = ser.apply(lambda x: np.nan if is_outlier(x, lower_bound, upper_bound) else x)

    # Ausgabe der Ausreißer und ihrer Indizes
    outlier_values = ser[~ser.isin(outliers)]
    print(f'Ausreißer-Werte: {outlier_values.dropna().tolist()}')
    outlier_indices = outlier_values.index
    print(f'Ausreißer-Indizes: {outlier_indices.tolist()}')
    num_outliers = len(outlier_values.dropna())
    print(f'Anzahl der Ausreißer: {num_outliers}')

    response = input(f'Sollen die Ausreißer ersetzt werden? (ja/nein): ')
    if response.lower() == 'ja':
        # Anzahl der NaN-Werte nach dem Setzen der Ausreißer zu NaN
        num_nans_after = outliers.isna().sum()
        print(f'Anzahl der NaN-Werte nach dem Setzen aller Ausreißer auf NaN: {num_nans_after}')

        # Interpolation nur mit Werten, die nicht 0 sind
        mask_non_zero = (ser != 0)  # Maske, um 0-Werte zu ignorieren
        ser_interpolated = outliers.where(mask_non_zero).interpolate(method='linear', limit_direction='both')

        # Für die verbleibenden NaN-Werte, die von 0-Werten umgeben sind, diese zurücksetzen
        ser_final = outliers.combine_first(ser_interpolated)

        return ser_final

    return ser
# -------------------------------------------------------------

# 4.5 Anwendung der Ausreißerkorrektur

(heizungswasser == 0).sum()
heizungswasser = process_outliers_000(heizungswasser)
strom = process_outliers(strom)
Kälte = process_outliers_000(Kälte)
# -------------------------------------------------------------


# 5 Darstellung der Energiebedarfe

# 5.1 Energiebedarfe für 2024 plotten

# Manuell berechnete Tagesmittelwerte
kaelte_daily = Kälte.values.reshape(-1, 24).mean(axis=1)
strom_daily = strom.values.reshape(-1, 24).mean(axis=1)
heizung_daily = heizungswasser.values.reshape(-1, 24).mean(axis=1)

# X-Achse: Tage von 1 bis 366
tage = np.arange(1, len(kaelte_daily) + 1)

# Neue xticks für Monatsnamen, z. B. nach ca. jedem 30. Tag
month_positions_day = [15, 45, 75, 105, 135, 165, 195, 225, 255, 285, 315, 345]  # ungefähre Tagesmitten
month_labels = ["Jan", "Feb", "Mrz", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


# Plot 1
kaelte_median = np.median(Kälte.values)
kaelte_max = np.max(Kälte.values)

plt.figure(figsize=(15, 5))
plt.plot(tage, kaelte_daily,  label=f'Kältebedarf [kW] (Median: {kaelte_median:.1f} kW, Max: {kaelte_max:.1f} kW)', color=color_kälte)
plt.fill_between(tage, kaelte_daily, color=color_kälte, alpha=0.3)  # Füllt die Fläche
plt.title('Täglicher Kältebedarf 2024')
plt.xlabel('Monat')
plt.ylabel('Kältebedarf [kW]')
plt.grid(True)
plt.xticks(month_positions_day, month_labels, fontsize=11)
plt.tight_layout()
plt.legend()
plt.savefig('Kältebedarf_2024_Tagesmittel_ohne_Index.jpg')

# Plot 2
strom_median = np.median(strom.values)
strom_max = np.max(strom.values)

plt.figure(figsize=(15, 5))
plt.plot(tage, strom_daily, label=f'Strombedarf [kW] (Median: {strom_median:.1f} kW, Max: {strom_max:.1f} kW)', color=color_strom)
plt.fill_between(tage, strom_daily, color=color_strom, alpha=0.3)  # Füllt die Fläche
plt.title('Täglicher Strombedarf 2024')
plt.xlabel('Monat')
plt.ylabel('Strombedarf [kW]')
plt.grid(True)
plt.xticks(month_positions_day, month_labels, fontsize=11)
plt.tight_layout()
plt.legend()
plt.savefig('Strombedarf_2024_Tagesmittel_ohne_Index.jpg')

# Plot 3
heizung_median = np.median(heizungswasser.values)
heizung_max = np.max(heizungswasser.values)

plt.figure(figsize=(15, 5))
plt.plot(tage, heizung_daily, label=f'Wärmebedarf [kW] (Median: {heizung_median:.1f} kW, Max: {heizung_max:.1f} kW)', color=color_raumwaerme)
plt.fill_between(tage, heizung_daily, color=color_raumwaerme, alpha=0.3)  # Füllt die Fläche
plt.title('Täglicher Wärmebedarf 2024')
plt.xlabel('Monat')
plt.ylabel('Wärmebedarf [kW]')
plt.grid(True)
plt.xticks(month_positions_day, month_labels, fontsize=11)
plt.tight_layout()
plt.legend()
plt.savefig('Wärmebedarf_2024_Tagesmittel_ohne_Index.jpg')
plt.show()
# -------------------------------------------------------------


# 5.2 Modellierung Strompreise

month_positions = [
    372,    # Januar
    1116,   # Februar
    1788,   # März
    2520,   # April
    3264,   # Mai
    3972,   # Juni
    4716,   # Juli
    5456,   # August
    6192,   # September
    6936,   # Oktober
    7670,   # November
    8424    # Dezember
]
month_labels = ["Jan", "Feb", "Mrz", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

print(df_strom_pr_em_2024.columns)

df_strom_pr_em_2024["Strompreis"].mean()

# ursprüngliche Agra Strompreise
plt.figure(figsize=(15, 5))
plt.plot(df_strom_pr_em_2024["Strompreis"], label='Strompreis Agora', color=color_raumwaerme)
plt.title('Strompreis über den Jahresverlauf 2024')
plt.xlabel('Monat')
plt.ylabel('Strompreis in €/MWh')
plt.xticks(month_positions, month_labels, fontsize=11)

# Verschiebung der Strompreise in 2024

agora_mean=df_strom_pr_em_2024["Strompreis"].mean()
df_strom_pr_em_2024["Strompreis"] += 254.2-agora_mean
#df_strom_pr_em_2024["Strompreis"] += 189.2-agora_mean


#df_strom_pr_em_2024.loc[df_strom_pr_em_2024['Strompreis'] > 800, 'Strompreis'] -= 600    # Davor 206
df_strom_pr_em_2024['Strompreis'] = df_strom_pr_em_2024['Strompreis'].clip(upper=800)
plt.plot(df_strom_pr_em_2024["Strompreis"], label='Strompreis Grünenthal', color=color_strom)
print(f'Der Mittelert ist: {df_strom_pr_em_2024["Strompreis"].mean()}')
plt.grid(True)
plt.legend()
plt.savefig("Strompreis_über_den_Jahresverlauf_2024.svg", bbox_inches="tight")
plt.show()


strompreis_cost = df_strom_pr_em_2024['Strompreis'].apply(lambda x: x if x >= 0 else 0)
strompreis_revenue = df_strom_pr_em_2024['Strompreis'].apply(lambda x: abs(x) if x < 0 else 0)

# Anpassung Stromemissionen Co2 neutraler Strom als Möglichkeit:

df_strom_pr_em_2024["CO₂-Emissionsfaktor des Strommix"]=0
stromemissionen = df_strom_pr_em_2024["CO₂-Emissionsfaktor des Strommix"]
strompreis_cost = strompreis_cost/1000 # €/MWh in €/kWh
strompreis_revenue = strompreis_revenue/1000


# -------------------------------------------------------------

# 5.3 Modellierung der Preisschwankung +- 20 % des Strompreises


decrease_factor=0.8
increase_factor=1.2
df_strom_pr_em_2024["Szenario Strompreis sinkt"] = df_strom_pr_em_2024["Strompreis"]*decrease_factor
df_strom_pr_em_2024["Szenario Strompreis steigt"] = df_strom_pr_em_2024["Strompreis"]*increase_factor

plt.figure(figsize=(15, 5))
plt.plot(df_strom_pr_em_2024["Szenario Strompreis sinkt"], 'r', label='Szenario Strompreis sinkt')
plt.plot(df_strom_pr_em_2024["Szenario Strompreis steigt"], 'b', label='Szenario Strompreis steigt')
plt.plot(df_strom_pr_em_2024["Strompreis"], 'g', label='Strompreis Grünenthal 2024')
plt.ylim(0,1000)
plt.grid(True)
plt.xticks(month_positions, month_labels, fontsize=11)
plt.legend()
plt.show()

df_strom_pr_em_2024.to_pickle("StromPreisEmissionenSensi.pkl")

strompreis_cost_sinkt = df_strom_pr_em_2024['Szenario Strompreis sinkt'].apply(lambda x: x if x >= 0 else 0)
strompreis_revenue_sinkt = df_strom_pr_em_2024['Szenario Strompreis sinkt'].apply(lambda x: abs(x) if x < 0 else 0)
strompreis_cost_sinkt = strompreis_cost_sinkt/1000 # €/MWh in €/kWh
strompreis_revenue_sinkt = strompreis_revenue_sinkt/1000

strompreis_cost_steigt = df_strom_pr_em_2024['Szenario Strompreis steigt'].apply(lambda x: x if x >= 0 else 0)
strompreis_revenue_steigt = df_strom_pr_em_2024['Szenario Strompreis steigt'].apply(lambda x: abs(x) if x < 0 else 0)
strompreis_cost_steigt = strompreis_cost_steigt/1000 # €/MWh in €/kWh
strompreis_revenue_steigt = strompreis_revenue_steigt/1000

strompreis_cost_sinkt.to_pickle("strompreis_cost_sinkt.pkl")
strompreis_revenue_sinkt.to_pickle("strompreis_revenue_sinkt.pkl")

strompreis_cost_steigt.to_pickle("strompreis_cost_steigt.pkl")
strompreis_revenue_steigt.to_pickle("strompreis_revenue_steigt.pkl")
# -------------------------------------------------------------

# 5.4 Modellierung des Strompreises als Projektion des Strompreisszenarios 2030


# Schritt 1: Z-Transformation der Originalverteilung
strom_orig = df_strom_pr_em_2024['Strompreis']
mean_orig = strom_orig.mean()
std_orig = strom_orig.std()
z_scores = (strom_orig - mean_orig) / std_orig

# Schritt 2: Transformation mit neuem Mittelwert und Standardabweichung
mean_target = 250  # Ziel-Mittelwert in €/MWh
std_target = 100    # Ziel-Standardabweichung in €/MWh
df_strom_pr_em_2024['Szenario Strompreis 2030'] = z_scores * std_target + mean_target

# Schritt 3: Negative Strompreise ausschließen
df_strom_pr_em_2024.loc[df_strom_pr_em_2024['Szenario Strompreis 2030'] > 800, 'Szenario Strompreis 2030'] -= 500    # Davor 206
df_strom_pr_em_2024['Szenario Strompreis 2030'] = df_strom_pr_em_2024['Szenario Strompreis 2030'].clip(upper=800)

plt.figure(figsize=(6, 5))
plt.hist(df_strom_pr_em_2024['Strompreis'] / 1000, bins=175, label='Realer Strompreis 2024', color='tab:blue')
plt.hist(df_strom_pr_em_2024['Szenario Strompreis 2030'] / 1000, bins=175, label='Szenario Strompreis 2030', color='gray', alpha=0.6)
plt.xlabel("Strompreis in €/kWh")
plt.ylabel("Häufigkeit")
plt.legend()
plt.tight_layout()
plt.show()

werte = df_strom_pr_em_2024['Szenario Strompreis 2030']  # Beispiel

# Histogramm erzeugen
counts, bin_edges = np.histogram(werte, bins=175)

# Index des maximalen Werts im Histogramm
max_index = np.argmax(counts)

# Häufigster Wert (Modus) liegt in diesem Intervall:
häufigster_wert = (bin_edges[max_index] + bin_edges[max_index + 1]) / 2

print(f"Häufigster Wert im Histogramm: {häufigster_wert:.2f} €/MWh")


# Visualisierung der Verteilungen vor und nach der Transformation

fig, ax = plt.subplots(1, 2, figsize=(16/2.54, 12/2.54))

ax[0].hist(df_strom_pr_em_2024['Strompreis'], bins=175, color=blau_stromkauf)
ax[0].set_title('Vor der Transformation')
ax[0].set_xlabel('Strompreis [€/MWh]')
ax[0].set_ylabel('Häufigkeit')
ax[0].set_ylim(0, 850)

ax[1].hist(df_strom_pr_em_2024['Szenario Strompreis 2030'], bins=175, color=blau_stromkauf)
ax[1].set_title('Nach der Transformation')
ax[1].set_xlabel('Transformierter Strompreis [€/MWh]')
ax[1].set_ylabel('Häufigkeit')
ax[1].set_ylim(0, 850)

plt.tight_layout()
#plt.savefig("StrompreisTransformiert.pdf")
plt.show()


print(df_strom_pr_em_2024["Szenario Strompreis 2030"].mean())
def german_decimal_formatter(x, _):
    return f"{x:.1f}".replace('.', ',')

fig, ax = plt.subplots(figsize=(16/2.54, 12/2.54))

# Histogramm für den originalen Strompreis
ax.hist(df_strom_pr_em_2024['Strompreis']/1000, bins=175, label='Realer Strompreis 2024', color=blau_stromkauf)

# Histogramm für den transformierten Strompreis
ax.hist(df_strom_pr_em_2024['Szenario Strompreis 2030']/1000, bins=175, label='Szenario Strompreis 2030', color=(8/255, 8/255, 8/255), alpha=0.5)

ax.set_xlabel('Strompreis in €/kWh', fontsize=11)
ax.set_ylabel('Häufigkeit', fontsize=11)
ax.set_ylim(0, 500)
ax.legend(loc="upper left", fontsize=11)
ax.xaxis.set_major_formatter(FuncFormatter(german_decimal_formatter))

plt.gca().xaxis.set_major_formatter(formatter)

plt.tight_layout()
plt.savefig("Abb5_20_StrompreisTransformiert.pdf")
plt.show()


color_strom = (0/255, 83/255, 116/255)  # RGB (0, 83, 116)
color_raumwaerme = (176/255, 0/255, 70/255)  # RGB (176, 0, 70)
color_raumwaerme_light = (243/255, 217/255, 227/255)
custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme", [color_strom, color_raumwaerme_light, color_raumwaerme])

strompreis_matrix = df_strom_pr_em_2024['Strompreis'].values.reshape((366, 24))
strompreis_matrix = strompreis_matrix.T

strompreis_matrix_2 = df_strom_pr_em_2024['Szenario Strompreis 2030'].values.reshape((366, 24))
strompreis_matrix_2 = strompreis_matrix_2.T

fig, ax = plt.subplots(1, 2, figsize=(16/2.54 * 2, 12/2.54))

# Linke Heatmap für "Strompreis 2023"
sns.heatmap(strompreis_matrix, cmap=custom_cmap_blaurot, annot=False, fmt=".2f",
            yticklabels=range(24), xticklabels=range(1, 366), cbar_kws={'label': 'Strompreis [€/MWh]'}, ax=ax[0])

# Customize x-ticks: Show every 50st day
ax[0].set_xticks(np.arange(0, strompreis_matrix.shape[1], 50))
ax[0].set_xticklabels(np.arange(0, strompreis_matrix.shape[1] + 1, 50), fontsize=11, rotation=0)

# Customize y-ticks: Show every third hour
ax[0].set_yticks(np.arange(0, 24, 3))
ax[0].set_yticklabels(np.arange(0, 24, 3), fontsize=11)

ax[0].set_xlabel('Tag des Jahres', fontsize=12)
ax[0].set_ylabel('Stunde des Tages', fontsize=12)
ax[0].set_title('Stündlicher Strompreis', fontsize=13)
ax[0].invert_yaxis()

# Rechte Heatmap für "Strompreis 2030"
sns.heatmap(strompreis_matrix_2, cmap=custom_cmap_blaurot, annot=False, fmt=".2f",
            yticklabels=range(24), xticklabels=range(1, 366), cbar_kws={'label': 'Stündlicher Transformierter Strompreis [€/MWh]'}, ax=ax[1])

# Customize x-ticks: Show every 50st day
ax[1].set_xticks(np.arange(0, strompreis_matrix_2.shape[1], 50))
ax[1].set_xticklabels(np.arange(0, strompreis_matrix_2.shape[1] + 1, 50), fontsize=11, rotation=0)

# Customize y-ticks: Show every third hour
ax[1].set_yticks(np.arange(0, 24, 3))
ax[1].set_yticklabels(np.arange(0, 24, 3), fontsize=11)

ax[1].set_xlabel('Tag des Jahres', fontsize=12)
ax[1].set_ylabel('Stunde des Tages', fontsize=12)
ax[1].set_title('Stündlicher Transformierter Strompreis', fontsize=13)
ax[1].invert_yaxis()

plt.tight_layout()
plt.show()

strompreis_cost_schwankt = df_strom_pr_em_2024['Szenario Strompreis 2030'].apply(lambda x: x if x >= 0 else 0)
strompreis_revenue_schwankt = df_strom_pr_em_2024['Szenario Strompreis 2030'].apply(lambda x: abs(x) if x < 0 else 0)
strompreis_cost_schwankt = strompreis_cost_schwankt/1000 # €/MWh in €/kWh
strompreis_revenue_schwankt = strompreis_revenue_schwankt/1000

strompreis_cost_schwankt.to_pickle("strompreis_cost_schwankt.pkl")
strompreis_revenue_schwankt.to_pickle("strompreis_revenue_schwankt.pkl")
# -------------------------------------------------------------

# 5.5 Modellierung des Gaspreises


total_hours = 8784

# Erstelle eine Liste von Werten, im ersten Halbjahr 0.0771 EUR/kWh und im zweiten Halbjahr 0.0868 (Stand 2024)
halbjahreswerte_2024 = [0.07 if i < 4345 else 0.0774 for i in range(total_hours)]

# Erstelle den DataFrame mit einem Index von 1 bis 8760
gaspreis = pd.Series(halbjahreswerte_2024, index=range(0, total_hours))

halbjahreswerte_2024_sensitivitätsanalyse = [0.0336 for i in range(total_hours)]
gaspreis_sensi = pd.Series(halbjahreswerte_2024_sensitivitätsanalyse, index=range(0, total_hours))

strom.to_pickle("stromverbrauch.pkl")
heizungswasser.to_pickle("heizungswasser.pkl")
#Prozesswaerme.to_pickle("prozesswärmebedarf.pkl")
strompreis_cost.to_pickle("strompreis_cost.pkl")
strompreis_revenue.to_pickle("strompreis_revenue.pkl")
stromemissionen.to_pickle("stromemissionen.pkl")
gaspreis.to_pickle("gaspreis.pkl")
gaspreis_sensi.to_pickle("gaspreisSensi.pkl")
Kälte.to_pickle("kältebedarf.pkl")
# -------------------------------------------------------------
# 6. Zählerstände

# 6.1  Darstellung Zählerstände 2024

categories = ['Gasverbrauch Geb. 204 Einspeisung W5 - ∅ - kW', 'Gasverbrauch Geb. 307 BHKW - ∅ - kW', 'Stromzähler Geb. 307 BHKW Generator Einspeisung - ∅ - kW', 'Stromzähler W5 20kV-Einspeisung Kaubendenstr. (Geb. 809) - ∅ - kW',
              'Wärmeerzeugung BHKW Geb. 307 - ∅ - kW', 'Heizungswasser Geb. 307 Notkühler - ∅ - kW', 'Heizungswassererzeugung Geb. 307 BHKW (ohne Notkühler) - ∅ - kW',
              'Heißwassererzeugung Geb. 307 gesamt - ∅ - kW', 'Heißwassererzeugung Geb. 307 aus BHKW - ∅ - kW', 'Heizungswasser von BHKW307-Netz zu Geb. 201 301 407 - ∅ - kW',
              'Kaltwasser Geb. 307 gesamt - ∅ - kW', 'Heizungswasser Geb. 307 Absorbtionskälte - ∅ - kW']

capacities = [df_2024["Gasverbrauch Geb. 204 Einspeisung W5 - ∅ - kW"].sum(),df_bhkw_2024["Gasverbrauch Geb. 307 BHKW - ∅ - kW"].sum(),
              df_bhkw_2024["Stromzähler Geb. 307 BHKW Generator Einspeisung - ∅ - kW"].sum(),df_2024["Stromzähler W5 20kV-Einspeisung Kaubendenstr. (Geb. 809) - ∅ - kW"].sum(),
              df_bhkw_2024["Wärmeerzeugung BHKW Geb. 307 - ∅ - kW"].sum(), df_bhkw_2024["Heizungswasser Geb. 307 Notkühler - ∅ - kW"].sum(),
              df_bhkw_2024["Heizungswassererzeugung Geb. 307 BHKW (ohne Notkühler) - ∅ - kW"].sum(),df_bhkw_2024["Heißwassererzeugung Geb. 307 gesamt - ∅ - kW"].sum(),
              df_bhkw_2024["Heißwassererzeugung Geb. 307 aus BHKW - ∅ - kW"].sum(), df_bhkw_2024["Heizungswasser von BHKW307-Netz zu Geb. 201 301 407 - ∅ - kW"].sum(),
              process_outliers(df_bhkw_2024["Kaltwasser Geb. 307 gesamt - ∅ - kW"]).sum(), df_bhkw_2024["Heizungswasser Geb. 307 Absorbtionskälte - ∅ - kW"].sum()]

capacities = [x / 1e6 for x in capacities]

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

plt.figure(figsize=(20, 18))

bars = plt.barh(categories, capacities, color=[color_strom])

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.05*max(capacities), bar.get_y() + bar.get_height()/2,
             f'{width:,.2f}'.replace(",", "."), va='center', fontsize=18)

plt.xlabel('Summe 2024 in GWh', fontsize=20)
plt.title('relevante Zählerstände', fontsize=22)
plt.xlim(0, max(capacities) * 1.2)  # etwas Luft rechts
plt.gca().xaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(f"relevante Zählerstände.jpg", bbox_inches='tight')
plt.show()
# -------------------------------------------------------------


# 6.2 Untersuchung BHKW

gastest1 = df_bhkw_2024["Gasverbrauch Geb. 307 BHKW - ∅ - kW"]#*0.36
gastest2 = df_2024["Gasverbrauch Geb. 204 Einspeisung W5 - ∅ - kW"]#*0.8493

gastest1 = process_outliers_000(gastest1)   # Ergänzung zum bisherigen Code
gastest2 = process_outliers(gastest2)   # Ergänzung zum bisherigen Code

plt.figure(figsize=(16/2.54, 12/2.54))
plt.plot(gastest1, label="Gasverbrauch Geb. 307 BHKW")
plt.plot(gastest2, label="Gasverbrauch Geb. 204 BHKW+Heizwasserkessel")
plt.plot(strom, label="Strombedarf")
plt.legend(fontsize=11)
plt.xlabel("Stunden des Jahres", fontsize=12)
plt.xticks(fontsize=11)
plt.ylabel("kW", fontsize=12)
plt.title("Auffällige BHKW-Zeitreihe", fontsize=13)
plt.show()

gastest1.sum()
gastest2.sum()

#bhkw_gas_in = df_bhkw_2024["Gasverbrauch Geb. 307 BHKW - ∅ - kW"]                                          # Anpassung, da Daten aus 2024 fehlerhaft
bhkw_gas_in = df_2024["Gasverbrauch Geb. 204 Einspeisung W5 - ∅ - kW"] # inklusive Heizwasserkessel
bhkw_heizungswasser_out = df_bhkw_2024["Heizungswasser von BHKW307-Netz zu Geb. 201 301 407 - ∅ - kW"] + df_bhkw_2024["Heizungswasser Geb. 307 Absorbtionskälte - ∅ - kW"] + df_bhkw_2024["Heizungswasser Geb. 307 Notkühler - ∅ - kW"] + df_bhkw_2024["Heizungswasser Geb. 307 gesamt - ∅ - kW"]
bhkw_heizungswasser_out_2 = df_bhkw_2024["Heizungswassererzeugung Geb. 307 BHKW (ohne Notkühler) - ∅ - kW"] + df_bhkw_2024["Heizungswasser Geb. 307 Notkühler - ∅ - kW"]
bhkw_strom_out = df_bhkw_2024["Stromzähler Geb. 307 BHKW Generator Einspeisung - ∅ - kW"] + df_2024["Stromzähler W5 20kV-Ausspeisung Kaubendenstr. (Geb. 809) - ∅ - kW"] # Stromüberschuss/Netzeinspeisung ist NICHT irrelevant # wärmegeführt
bhkw_heisswasser_out = df_bhkw_2024["Heißwassererzeugung Geb. 307 aus BHKW - ∅ - kW"]
bhkw_wärme_out = bhkw_heizungswasser_out + bhkw_heisswasser_out

(bhkw_gas_in == 0).sum()
bhkw_gas_in = process_series(bhkw_gas_in)
bhkw_gas_in = process_outliers_000(bhkw_gas_in)

bhkw_wärme_out = process_outliers_000(bhkw_wärme_out)
bhkw_heizungswasser_out = process_series(bhkw_heizungswasser_out)
bhkw_heizungswasser_out = process_outliers_000(bhkw_heizungswasser_out)
bhkw_heizungswasser_out_2 = process_series(bhkw_heizungswasser_out_2)
bhkw_strom_out = process_series(bhkw_strom_out)
bhkw_heisswasser_out = process_series(bhkw_heisswasser_out)
bhkw_heisswasser_out = process_outliers_000(bhkw_heisswasser_out)

bhkw_heizungswasser_out.sum()
bhkw_heizungswasser_out.sum()

plt.figure(figsize=(15,7))
plt.plot(bhkw_heizungswasser_out, color="blue")
plt.plot(bhkw_heizungswasser_out_2, color="red")
plt.show()

bhkw_energiestroeme = pd.DataFrame({
    'bhkw_gas_in': bhkw_gas_in,
    'bhkw_heizungswasser_out': bhkw_heizungswasser_out,
    'bhkw_strom_out': bhkw_strom_out,
    'bhkw_heisswasser_out': bhkw_heisswasser_out,
    'bhkw_wärme_out': bhkw_wärme_out
})
gasverbrauch_BHKW=bhkw_gas_in.sum()/(1000 * 1000)
#gasverbrauch_BHKW=gasverbrauch_BHKW.replace('.', ',')
bhkw_energiestroeme_melted = bhkw_energiestroeme.melt(var_name='Type', value_name='kWh')

plt.figure(figsize=(25/2.54, 12/2.54))
sns.violinplot(x='Type', y='kWh', data=bhkw_energiestroeme_melted, cut=0)
plt.xlabel('Energiefluss', fontsize=12)
plt.xticks(rotation=45, fontsize=11)
plt.ylabel('kWh', fontsize=12)
plt.yticks(fontsize=11)
plt.title(f'Verteilung der BHKW-Energieströme mit Gasverbrauch von {gasverbrauch_BHKW} GWh', fontsize=13)

plt.tight_layout()
plt.show()

eta_heizungswasser = bhkw_energiestroeme['bhkw_heizungswasser_out'].mean() / bhkw_energiestroeme['bhkw_gas_in'].mean()
eta_strom = bhkw_energiestroeme['bhkw_strom_out'].mean() / bhkw_energiestroeme['bhkw_gas_in'].mean()
eta_heisswasser = bhkw_energiestroeme['bhkw_heisswasser_out'].mean() / bhkw_energiestroeme['bhkw_gas_in'].mean()
th_sum = eta_heisswasser + eta_heizungswasser
eta_wärme = bhkw_energiestroeme["bhkw_wärme_out"].mean() / bhkw_energiestroeme["bhkw_gas_in"].mean()
print(f"eta_wärme_gesamt: {eta_wärme: .4f}")
print(f"eta_th: {th_sum:.4f}")
print(f"eta_el: {eta_strom:.4f}")
total_sum = eta_heizungswasser + eta_heisswasser + eta_strom
print(f"sum: {eta_heizungswasser:.4f} + {eta_heisswasser:.4f} + {eta_strom:.4f} = {total_sum:.4f}")

eta_heizungswasser = bhkw_energiestroeme['bhkw_heizungswasser_out'].sum() / bhkw_energiestroeme['bhkw_gas_in'].sum()
eta_strom = bhkw_energiestroeme['bhkw_strom_out'].sum() / bhkw_energiestroeme['bhkw_gas_in'].sum()
eta_heisswasser = bhkw_energiestroeme['bhkw_heisswasser_out'].sum() / bhkw_energiestroeme['bhkw_gas_in'].sum()
th_sum = eta_heisswasser + eta_heizungswasser
eta_wärme = bhkw_energiestroeme["bhkw_wärme_out"].sum() / bhkw_energiestroeme["bhkw_gas_in"].sum()
print(f"eta_wärme_gesamt: {eta_wärme: .4f}")
print(f"eta_th: {th_sum:.4f}")
print(f"eta_el: {eta_strom:.4f}")
total_sum = eta_heizungswasser + eta_heisswasser + eta_strom
print(f"sum: {eta_heizungswasser:.4f} + {eta_heisswasser:.4f} + {eta_strom:.4f} = {total_sum:.4f}")

bhkw_energiestroeme['eta_heizungswasser'] = bhkw_energiestroeme['bhkw_heizungswasser_out'] / bhkw_energiestroeme['bhkw_gas_in']
bhkw_energiestroeme['eta_strom'] = bhkw_energiestroeme['bhkw_strom_out'] / bhkw_energiestroeme['bhkw_gas_in']
bhkw_energiestroeme['eta_heisswasser'] = bhkw_energiestroeme['bhkw_heisswasser_out'] / bhkw_energiestroeme['bhkw_gas_in']
bhkw_energiestroeme['sum'] = bhkw_energiestroeme["eta_heisswasser"] + bhkw_energiestroeme["eta_heizungswasser"] + bhkw_energiestroeme["eta_strom"]

eta_heisswasser_test = bhkw_energiestroeme[bhkw_energiestroeme["eta_heisswasser"] < 1]["eta_heisswasser"].mean()
eta_heizungswasser_test = bhkw_energiestroeme[bhkw_energiestroeme["eta_heizungswasser"] < 1]["eta_heizungswasser"].mean()
eta_strom_test = bhkw_energiestroeme[bhkw_energiestroeme["eta_strom"] < 1]["eta_strom"].mean()
print(eta_heisswasser_test)
print(eta_heizungswasser_test)
print(eta_strom_test)

print(bhkw_energiestroeme["eta_heisswasser"].mean())
print(bhkw_energiestroeme["eta_heizungswasser"].mean())
print(bhkw_energiestroeme["eta_strom"].mean())

sinaplot_data = bhkw_energiestroeme[['eta_heizungswasser', 'eta_strom', 'eta_heisswasser', 'sum']]
sinaplot_data_melted = sinaplot_data.melt(var_name='Type', value_name='Value')

plt.figure(figsize=(16/2.54, 12/2.54))
sns.violinplot(x='Type', y='Value', data=sinaplot_data_melted, inner=None, color=".8")
sns.stripplot(x='Type', y='Value', data=sinaplot_data_melted, jitter=True, size=4)
plt.xlabel('Energiestrom', fontsize=12)
plt.ylabel('Wirkungsgrad [-]', fontsize=12)
plt.title('Umwandlungseffizienzen im BHKW', fontsize=13)
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.gca().yaxis.set_major_formatter(formatter)
plt.tight_layout()
plt.show()

allokationsfaktor_raumwärme = heizungswasser.sum() / bhkw_wärme_out.sum()
allokationsfaktor_strom = allokationsfaktor_raumwärme # ist nicht 1, weil nur wenn Gas für Raumwärme erzeugt wird, wird auch Strom erzeugt
allokationsfaktor_gasemissionen = (allokationsfaktor_raumwärme + allokationsfaktor_strom) / 2
print(allokationsfaktor_gasemissionen)
print(allokationsfaktor_raumwärme)
# -------------------------------------------------------------

# 7 Modellierung der PV-Leistung

# 7.1 Datenaufbereitung der PV-Leistung


path_pv = os.path.join(cd, "Timeseries_50.768_6.151_SA3_2000kWp_crystSi_14_39deg_-3deg_2023_2024.csv")
df_pv_2024 = pd.read_csv(path_pv, skiprows=10)

units = {
    "time": "UTC",
    "PV power output": "W",
    "Global in-plane irradiance": "W/m2",
    "Sun height": "°",
    "Air temperature": "ºC",
    "Int": "unknown"
}
df_pv_2024.columns = [f'{col} ({units[col]})' if col in units else col for col in df_pv_2024.columns]
print(df_pv_2024)
df_pv_2024.index = range(0, len(df_pv_2024))
print(df_pv_2024.columns.tolist())

pv = df_pv_2024['P']
print(f'Die gesamte PV-Leistung beträgt {pv.sum()/1000000}')

pv_skaliert=pv/1000
pv_nennleistung = 2000 # kW_p
pv_normalisiert = pv_skaliert/pv_nennleistung
df_pv_2024['pv_scaled (kW)'] = pv_skaliert
df_pv_2024['time (UTC)'] = pd.to_datetime(df_pv_2024['time (UTC)'], format='%Y%m%d:%H%M')
df_pv_2024.set_index('time (UTC)', inplace=True)
pv_hourly = df_pv_2024['pv_scaled (kW)'].resample('h').mean()
# -------------------------------------------------------------

# 7.2 Plotten der PV-Leistung


# Create a heatmap matrix
pv_matrix = pv_hourly.values.reshape(-1, 24).T

color_raumwaerme = (176/255, 0/255, 70/255)  # RGB (176, 0, 70)
color_raumwaerme_light = (243/255, 217/255, 227/255)
custom_cmap_rot = LinearSegmentedColormap.from_list("weiß_zu_raumwaerme", [color_raumwaerme_light, color_raumwaerme])

# PV in MWh pro Tag
pv_täglich = (df_pv_2024["P"] / 1_000_000).resample('D').sum()
jahressumme = pv_täglich.sum()

# Dynamische Monatsmarkierungen
month_positions = pd.date_range(start=pv_täglich.index.min(), end=pv_täglich.index.max(), freq='MS')
month_labels = [d.strftime('%b') for d in month_positions]  # Jan, Feb, ...

plt.figure(figsize=(12, 4))
plt.plot(pv_täglich, label=f'PV-Erzeugung [MWh] – ∑ {jahressumme:.0f} MWh', color='orange')
plt.fill_between(pv_täglich.index, pv_täglich.values, color='orange', alpha=0.3)

plt.xticks(month_positions, month_labels, fontsize=11)
plt.title("Tägliche mittlere PV-Leistung")
plt.xlabel("Datum")
plt.ylabel("Erzeugung [MWh/Tag]")

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', '.')))

plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# Create the heatmap
plt.figure(figsize=(16/2.54, 12/2.54))
ax = sns.heatmap(pv_matrix, cmap=custom_cmap_rot, cbar_kws={'label': 'PV-Erzeugung [kW]'})

# Customize x-ticks: Show every 50st day
ax.set_xticks(np.arange(0, pv_matrix.shape[1], 50))
ax.set_xticklabels(np.arange(0, pv_matrix.shape[1], 50), fontsize=11, rotation=0)

# Customize y-ticks: Show every third hour
ax.set_yticks(np.arange(0, 24, 3))
ax.set_yticklabels(np.arange(0, 24, 3), fontsize=11)

plt.xlabel('Tag des Jahres', fontsize=12)
plt.ylabel('Stunde des Tages', fontsize=12)

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=11)
cbar.set_label('PV-Anlagenleistung in kW', fontsize=12)

#plt.savefig("PVLeistung.pdf", format="pdf")

plt.show()

pv_normalisiert.to_pickle("pv.pkl")
# -------------------------------------------------------------


# 7.3 Preis für PV-Contracting
index = pd.date_range(start='2024-01-01', end='2024-12-31 23:00:00', freq='h')

pv_contracting=pd.DataFrame(index=index)
pv_contracting['Preis Contracting']=0.14

pv_contracting.to_pickle("pv_contracting.pkl")

# -------------------------------------------------------------

# 8 Modellierung der Wärmepumpe und Kompressionskältemaschine (KKM)

# 8.1  Modellierung des COP in Abhängigkeit der Außentemperatur


df = pd.read_excel("Temperatur_Aachen_2024_KORRIGIERT.xlsx")
df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"])
df.set_index("MESS_DATUM", inplace=True)

# Parameter für COP-Berechnung
eta = 0.55                    # Effizienzfaktor der Wärmepumpe
T_vorlauf_heizen = 90 + 273.15      # Vorlauftemperatur in Kelvin
T_vorlauf_kuelen=6+273.15
# Außentemperatur in Kelvin
T_außen = df["TT_TU_C"] + 273.15
df["TT_TU_C"].index = range(0, len(df["TT_TU_C"]))
df["TT_TU_C"].to_pickle("T_amb.pkl")

# COP berechnen (Formel vermeidet Division durch 0)
df["COP_Heizen"] = eta * T_vorlauf_heizen / (T_vorlauf_heizen - T_außen)
df["COP_Heizen"].index = range(0, len(df["COP_Heizen"]))

df["COP_Heizen"].to_pickle("cop.pkl")
df["COP_Heizen"].plot(figsize=(12, 4), title="COP der Wärmepumpe (Heizen) in Aachen 2024", ylabel="COP")
plt.grid(True)
plt.tight_layout()
plt.savefig("COP der Wärmepumpe (Heizen) in Aachen 2024.svg", bbox_inches="tight")
plt.show()

df["COP_Kühlen"] = eta * T_vorlauf_kuelen / (T_außen-T_vorlauf_kuelen)
df["COP_Kühlen"]=process_outliers_000(df["COP_Kühlen"])
df["COP_Kühlen"].plot(figsize=(12, 4), title="EER der Wärmepumpe (Kühlen) in Aachen 2024", ylabel="EER")
plt.grid(True)
plt.tight_layout()
plt.show()
# -------------------------------------------------------------
# 8.2 Versuch der Modellierung des Zusammenhangs zwischen PV-Bereitstellung und WP-Nutzung

print(heizungswasser.max()) # relevanter Auslegungswert

strom.max() # 1h Strom speichern
strom_agg = strom.groupby(strom.index // 4).sum() # über 4 Stunden Strom speichern können
print(strom_agg.max())
print(strom_agg.mean())
surplus = pv_skaliert - strom # PV-Strom speichern
print(surplus.max())

# Mittlere Stundenpositionen für jeden Monat in einem nicht-Schaltjahr (8760 Stunden)
month_positions = [
    372,    # Januar
    1116,   # Februar
    1788,   # März
    2520,   # April
    3264,   # Mai
    3972,   # Juni
    4716,   # Juli
    5456,   # August
    6192,   # September
    6936,   # Oktober
    7670,   # November
    8424    # Dezember
]
month_labels = ["Jan", "Feb", "Mrz", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]

def thousand_separator(x, _):
    return f"{x:,.0f}".replace(",", ".")

plt.figure(figsize=(16/2.54, 12/2.54))
plt.plot(surplus, color=(0/255, 83/255, 116/255))
plt.xticks(month_positions, month_labels, fontsize=11)
plt.ylabel("PV-Überschuss nach Bedarfsdeckung\nin kW", fontsize=12)
plt.yticks(fontsize=11)
plt.gca().yaxis.set_major_formatter(FuncFormatter(thousand_separator))
plt.tight_layout()
plt.savefig("Abb5_5_PVÜberschuss.pdf", format="pdf")
plt.show()

daily_surplus = surplus.groupby(surplus.index // 24).sum()

# Define seasons based on typical months
# Spring: March (59 days) - May (151 days)
# Summer: June (152 days) - August (243 days)
# Autumn: September (244 days) - November (334 days)
# Winter: December (335 days) - February (58 days)
spring_days = list(range(59, 152))
summer_days = list(range(152, 244))
autumn_days = list(range(244, 335))
winter_days = [i for i in range(len(daily_surplus)) if (i % 365 <= 58 or i % 365 >= 335)]

# ODER MAX?
# "nightly" surplus meint, dass der Strom nachts eingespeichert werden könnte, aber es handelt sich um den generell in 24h anfallenden Überschussstrom
def average_nightly_surplus(days):
    nightly_surplus = daily_surplus.iloc[days]
    average_nightly = nightly_surplus.apply(lambda x: x if x > 0 else 0).mean()
    return average_nightly

avg_nightly_spring = average_nightly_surplus(spring_days)
avg_nightly_summer = average_nightly_surplus(summer_days)
avg_nightly_autumn = average_nightly_surplus(autumn_days)
avg_nightly_winter = average_nightly_surplus(winter_days)

avg_nightly_storage = {
    'Spring': avg_nightly_spring,
    'Summer': avg_nightly_summer,
    'Autumn': avg_nightly_autumn,
    'Winter': avg_nightly_winter
}

avg_nightly_storage

heizungswasser.max() # 1h speichern können

heizungswasser_agg = heizungswasser.groupby(heizungswasser.index //24).sum() # 24h
print(heizungswasser_agg.max())
print(heizungswasser_agg.mean())

daily_heizungswasser = heizungswasser.groupby(heizungswasser.index //24).sum() # 24h nach Jahreszeiten

# Define seasons based on typical months
# Spring: March (59 days) - May (151 days)
# Summer: June (152 days) - August (243 days)
# Autumn: September (244 days) - November (334 days)
# Winter: December (335 days) - February (58 days)
spring_days = list(range(59, 152))
summer_days = list(range(152, 244))
autumn_days = list(range(244, 335))
winter_days = [i for i in range(len(daily_heizungswasser)) if (i % 365 <= 58 or i % 365 >= 335)]

def average_daily_heat_demand(days, daily_demand):
    total_daily_demand = daily_demand.iloc[days]
    return total_daily_demand.mean()

avg_daily_heat_spring = average_daily_heat_demand(spring_days, daily_heizungswasser)
avg_daily_heat_summer = average_daily_heat_demand(summer_days, daily_heizungswasser)
avg_daily_heat_autumn = average_daily_heat_demand(autumn_days, daily_heizungswasser)
avg_daily_heat_winter = average_daily_heat_demand(winter_days, daily_heizungswasser)

avg_daily_heat_demand = {
    'Spring': avg_daily_heat_spring,
    'Summer': avg_daily_heat_summer,
    'Autumn': avg_daily_heat_autumn,
    'Winter': avg_daily_heat_winter
}

avg_daily_heat_demand
# -------------------------------------------------------------

# 9 Visualisierung

# 9.1 Datenaufbereitung Energiebedarfe

Kältebedarf=Kälte.copy()
heizungswasserverbrauch = heizungswasser.copy()
#heisswasserverbrauch=Prozesswaerme.copy()
stromverbrauch = strom.copy()
datetime_index = pd.date_range(start='2024-01-01 01:00', end='2025-01-01 00:00', freq='h')
Kältebedarf.index= datetime_index
stromverbrauch.index = datetime_index
#heisswasserverbrauch.index=datetime_index
heizungswasserverbrauch.index = datetime_index
stromverbrauch.name = "Stromverbrauch"
heizungswasserverbrauch.name = "Raumwärme"
#heisswasserverbrauch.name="Prozesswärme/Verluste"
Kältebedarf.name = 'Kälte'


datetime_index = pd.date_range(start='2024-01-01 01:00', end='2025-01-01 00:00', freq='h')

#df_strom_pr_em_2024_plot = df_strom_pr_em[df_strom_pr_em.index.year == 2024]
df_strom_pr_em_2024_plot = df_strom_pr_em_2024[['Strompreis']]
df_strom_pr_em_2024['Strompreis'].index = datetime_index

df_strom_pr_em['Strompreis']=df_strom_pr_em_2024['Strompreis']

df_strom_pr_em_2024_plot = df_strom_pr_em[df_strom_pr_em.index.year == 2024]

print(type(df_strom_pr_em_2024_plot))
print(df_strom_pr_em_2024_plot.head())
print(df_strom_pr_em_2024_plot.columns if hasattr(df_strom_pr_em_2024_plot, 'columns') else "Keine Spalten vorhanden")

df_strom_pr_em_2024_plot.loc[:, "Strompreis"] = df_strom_pr_em_2024_plot["Strompreis"] / 1000  # €/MWh to €/kWh
gaspreis_vis = pd.DataFrame(gaspreis, columns=["Gaspreis"])
gaspreis_vis.index = datetime_index



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

# 9.2 Violinplot der Energiebedarfe

def plot_verbrauch(stromverbrauch, heizungswasserverbrauch, kältebedarf):
    # Jahresverbrauch berechnen und formatieren
    yearly_sum_mwh = {
        'Stromverbrauch [GWh]': stromverbrauch.sum() / (1000 * 1000),
        'Heizungswasserverbrauch [GWh]': heizungswasserverbrauch.sum() / (1000 * 1000),
        'Kältebedarf [GWh]': kältebedarf.sum()/(1000*1000)

    }

    stromverbrauch_gwh = f"{yearly_sum_mwh['Stromverbrauch [GWh]']:.2f}".replace('.', ',')
    heizungswasserverbrauch_gwh = f"{yearly_sum_mwh['Heizungswasserverbrauch [GWh]']:.2f}".replace('.', ',')
    kältebedarf_gwh = f"{yearly_sum_mwh['Kältebedarf [GWh]']:.2f}".replace('.', ',')

    def german_formatter(x, _):
        return f"{x:,.0f}".replace(",", ".")

    # Plot (a): Violinplot der Bedarfe
    plt.figure(figsize=(20 / 2.54, 10 / 2.54))
    sns.violinplot(data=[stromverbrauch, heizungswasserverbrauch, kältebedarf],
                   cut=0, palette=[color_strom, color_raumwaerme, color_kälte], inner="quart")
    plt.xticks([0, 1, 2], ['Strombedarf', 'Raumwärmebedarf', 'Kältebedarf'], fontsize=11)
    plt.ylabel("Leistung [kW}", fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(german_formatter))
    plt.title(
        "Jahressummen der Bedarfe \n"
        f"Strom: {stromverbrauch_gwh} GWh   Raumwärme: {heizungswasserverbrauch_gwh} GWh   Kälte: {kältebedarf_gwh} GWh",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("Abb5_1_StromHeizungswasserVerbrauch_a.pdf", format="pdf")

    # Plot (b): Monatliche Durchschnittswerte
    monthly_avg_strom = stromverbrauch.resample('MS').mean()
    monthly_avg_heizungswasser = heizungswasserverbrauch.resample('MS').mean()
    monthly_avg_kälte = kältebedarf.resample('MS').mean()


    plt.figure(figsize=(20 / 2.54, 10 / 2.54))
    plt.plot(monthly_avg_strom.index, monthly_avg_strom, label='Strombedarf', color=color_strom)
    plt.plot(monthly_avg_heizungswasser.index, monthly_avg_heizungswasser, label='Raumwärmebedarf', color=color_raumwaerme)
    plt.plot(monthly_avg_kälte.index, monthly_avg_kälte, label='Kälte', color=color_kälte)
    plt.legend(fontsize=11, loc="center right")
    plt.ylabel("Leistung in kW", fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(german_formatter))

    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b'))

    plt.tight_layout()
    plt.savefig("Abb5_1_StromHeizungswasserVerbrauch_b.pdf", format="pdf")

    # Plot (c): Wochendurchschnittswerte
    weekly_avg_strom = stromverbrauch.groupby(stromverbrauch.index.dayofweek).mean()
    weekly_avg_heizungswasser = heizungswasserverbrauch.groupby(heizungswasserverbrauch.index.dayofweek).mean()
    weekly_avg_kälte = kältebedarf.groupby(kältebedarf.index.dayofweek).mean()


    plt.figure(figsize=(20 / 2.54, 10 / 2.54))
    days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
    plt.plot(weekly_avg_strom.index, weekly_avg_strom, label='Strombedarf', color=color_strom)
    plt.plot(weekly_avg_heizungswasser.index, weekly_avg_heizungswasser, label='Raumwärmebedarf', color=color_raumwaerme)
    plt.plot(weekly_avg_kälte.index, weekly_avg_kälte, label='Kälte', color=color_kälte)
    plt.xticks(ticks=range(7), labels=days, fontsize=11)
    plt.legend(fontsize=11, loc="center right")
    plt.ylabel("Leistung in kW", fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(german_formatter))
    plt.ylim(0, 2200)

    plt.tight_layout()
    plt.savefig("Abb5_1_StromHeizungswasserVerbrauch_c.pdf", format="pdf")

plot_verbrauch(stromverbrauch, heizungswasserverbrauch, Kältebedarf)
# -------------------------------------------------------------

# 9.3 Violinplot des Strom- und Gaspreises

def plot_preis(gaspreis, strompreisemissionen):
    # Jahresdurchschnittspreise berechnen und formatieren
    yearly_avg_prices = {
        'Strompreis [€/kWh]': strompreisemissionen['Strompreis'].mean(),
        'Gaspreis [€/kWh]': gaspreis['Gaspreis'].mean(),
    }

    strompreis_avg = f"{yearly_avg_prices['Strompreis [€/kWh]']:.3f}".replace('.', ',')
    gaspreis_avg = f"{yearly_avg_prices['Gaspreis [€/kWh]']:.3f}".replace('.', ',')

    def german_formatter(x, _):
        return f"{x:.2f}".replace('.', ',')

    # Plot (a): Violinplot der Preise
    plt.figure(figsize=(16 / 2.54, 8 / 2.54))
    sns.violinplot(data=[
        strompreisemissionen["Strompreis"],
        gaspreis["Gaspreis"]],
        bw_adjust=0.1, palette=[color_strom, color_raumwaerme], inner="quart")
    plt.xticks([0, 1], ['Strompreis', 'Gaspreis'], fontsize=11)
    plt.ylabel("Preise in €/kWh", fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(german_formatter))
    plt.title(f"Jahresdurchschnitt Preise\n Strom: {strompreis_avg} €/kWh               Gas: {gaspreis_avg} €/kWh", fontsize=11)
    plt.tight_layout()
    plt.savefig("Abb5_3_PreisePlot_a.pdf", format="pdf")

    # Plot (b): Monatliche Durchschnittswerte
    monthly_avg_gas = gaspreis.resample('MS').mean()
    monthly_avg_strom = strompreisemissionen.resample('MS').mean()

    plt.figure(figsize=(16 / 2.54, 8 / 2.54))
    plt.plot(monthly_avg_strom.index, monthly_avg_strom['Strompreis'], label='Strompreis', color=color_strom)
    plt.plot(monthly_avg_gas.index, monthly_avg_gas['Gaspreis'], label='Gaspreis', color=color_raumwaerme)
    plt.ylabel("Preise in €/kWh", fontsize=11)
    plt.legend(fontsize=11, loc="upper right")
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(german_formatter))

    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b'))

    plt.tight_layout()
    plt.savefig("Abb5_3_PreisePlot_b.pdf", format="pdf")

    # Plot (c): Wochendurchschnittswerte
    weekly_avg_strom = strompreisemissionen.groupby(strompreisemissionen.index.dayofweek).mean()
    weekly_avg_gas = gaspreis.groupby(gaspreis.index.dayofweek).mean()

    plt.figure(figsize=(16 / 2.54, 8 / 2.54))
    days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
    plt.plot(weekly_avg_strom.index, weekly_avg_strom['Strompreis'], label='Strompreis', color=color_strom)
    plt.plot(weekly_avg_gas.index, weekly_avg_gas['Gaspreis'], label='Gaspreis', color=color_raumwaerme)
    plt.xticks(ticks=range(7), labels=days, fontsize=11)
    plt.ylabel("Preise in €/kWh", fontsize=11)
    plt.legend(fontsize=11, loc="upper right")
    plt.yticks(fontsize=11)
    #plt.ylim(0.067, 0.131)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(german_formatter))

    plt.tight_layout()
    plt.savefig("Abb5_3_PreisePlot_c.pdf", format="pdf")

plot_preis(gaspreis_vis, df_strom_pr_em_2024_plot)
# -------------------------------------------------------------

# 9.4 Heatmap des Strom- und Gaspreises


color_raumwaerme = (176/255, 0/255, 70/255)  # RGB (176, 0, 70)
color_raumwaerme_light = (243/255, 217/255, 227/255)
color_strom = (0/255, 83/255, 116/255)
color_strom_light = (217/255, 229/255, 234/255)
custom_cmap_rot = LinearSegmentedColormap.from_list("weiß_zu_raumwaerme", [color_raumwaerme_light, color_raumwaerme])
#custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme", [(124/255, 205/255, 239/255), (190/255, 30/255, 70/255)])
custom_cmap_blaurot = LinearSegmentedColormap.from_list("strom_zu_raumwaerme", [color_strom, color_raumwaerme_light, color_raumwaerme])

strompreis_matrix = df_strom_pr_em_2024['Strompreis'].values.reshape((366, 24))/1000
strompreis_matrix = strompreis_matrix.T

plt.figure(figsize=(16/2.54, 12/2.54))
ax = sns.heatmap(strompreis_matrix, cmap=custom_cmap_blaurot, annot=False, fmt=".2f", yticklabels=range(24), xticklabels=range(1, 366), cbar_kws={'label': 'Strompreis [€/kWh]'})

# Customize x-ticks: Show every 50st day
ax.set_xticks(np.arange(0, strompreis_matrix.shape[1], 50))
ax.set_xticklabels(np.arange(0, strompreis_matrix.shape[1] + 1, 50), fontsize=11, rotation=0)

# Customize y-ticks: Show every third hour
ax.set_yticks(np.arange(0, 24, 3))
ax.set_yticklabels(np.arange(0, 24, 3), fontsize=11)

plt.xlabel('Tag des Jahres', fontsize=12)
plt.ylabel('Stunde des Tages', fontsize=12)
plt.title('Stündlicher Strompreis', fontsize=13)

ax.invert_yaxis()

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=11)
cbar.set_label('€/kWh', fontsize=12)

#plt.savefig("StrompreisHeatmap.pdf", format="pdf")
# -------------------------------------------------------------


# 9.5 Violinplot für Strom- und Gasemissionen

gasemissionen = pd.DataFrame({'Gasemissionen [g/kWh]': [200.8] * 8760})
gasemissionen.index = pd.date_range(start='2024-01-01', periods=8760, freq='h')

def plot_emissionen(strompreisemissionen, gasemissionen):
    # Jahresdurchschnittswerte berechnen und formatieren
    yearly_sum = {
        'Stromemissionen [g/kWh]': strompreisemissionen["CO₂-Emissionsfaktor des Strommix"].mean(),
        "Gasemissionen [g/kWh]": gasemissionen["Gasemissionen [g/kWh]"].mean()
    }

    stromemission_avg = f"{yearly_sum['Stromemissionen [g/kWh]']:.1f}".replace('.', ',')
    gasemission_avg = f"{yearly_sum['Gasemissionen [g/kWh]']:.1f}".replace('.', ',')

    # Plot (a): Violinplot der Emissionen
    plt.figure(figsize=(16 / 2.54, 8 / 2.54))
    sns.violinplot(data=[
        strompreisemissionen["CO₂-Emissionsfaktor des Strommix"],
        gasemissionen["Gasemissionen [g/kWh]"]],
        cut=0, palette=[color_strom, color_raumwaerme], inner="quart")
    plt.xticks([0, 1], ['Stromemissionen', "Gasemissionen"], fontsize=11)
    plt.ylabel("Emissionen in g/kWh", fontsize=11)
    plt.yticks(fontsize=11)
    plt.title(f"Jahresdurchschnitt Emissionen\n Strom: {stromemission_avg} g/kWh               Gas: {gasemission_avg} g/kWh", fontsize=11)
    plt.tight_layout()
    plt.savefig("Abb5_4_EmissionenPlot_a.pdf", format="pdf")

    # Plot (b): Monatliche Durchschnittswerte
    monthly_avg_strompreisemissionen = strompreisemissionen.resample('MS').mean()
    monthly_avg_gasemissionen = gasemissionen.resample("MS").mean()

    plt.figure(figsize=(16 / 2.54, 8 / 2.54))
    plt.plot(monthly_avg_strompreisemissionen.index, monthly_avg_strompreisemissionen["CO₂-Emissionsfaktor des Strommix"],
             label='Stromemissionen', color=color_strom)
    plt.plot(monthly_avg_gasemissionen.index, monthly_avg_gasemissionen["Gasemissionen [g/kWh]"],
             label="Gasemissionen", color=color_raumwaerme)
    plt.legend(fontsize=11, loc="center right")
    plt.ylabel("Emissionen in g/kWh", fontsize=11)
    plt.ylim(190, 440)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b'))

    plt.tight_layout()
    plt.savefig("Abb5_4_EmissionenPlot_b.pdf", format="pdf")

    # Plot (c): Wochendurchschnittswerte
    weekly_avg_strompreisemissionen = strompreisemissionen.groupby(strompreisemissionen.index.dayofweek).mean()
    weekly_avg_gasemissionen = gasemissionen.groupby(gasemissionen.index.dayofweek).mean()

    plt.figure(figsize=(16 / 2.54, 8 / 2.54))
    days = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']
    plt.plot(weekly_avg_strompreisemissionen.index, weekly_avg_strompreisemissionen["CO₂-Emissionsfaktor des Strommix"],
             label='Stromemissionen', color=color_strom)
    plt.plot(weekly_avg_gasemissionen.index, weekly_avg_gasemissionen["Gasemissionen [g/kWh]"],
             label='Gasemissionen', color=color_raumwaerme)
    plt.xticks(ticks=range(7), labels=days, fontsize=11)
    plt.ylabel("Emissionen in g/kWh", fontsize=11)
    plt.legend(fontsize=11, loc="upper right")
    plt.ylim(190, 440)
    plt.yticks(fontsize=11)

    plt.tight_layout()
    plt.savefig("Abb5_4_EmissionenPlot_c.pdf", format="pdf")

plot_emissionen(df_strom_pr_em_2024_plot, gasemissionen)
# -------------------------------------------------------------


# 9.6 Violinplot für Strom- und Gasemissionen

stromemissionen_matrix = df_strom_pr_em_2024['CO₂-Emissionsfaktor des Strommix'].values.reshape((366, 24))
stromemissionen_matrix = stromemissionen_matrix.T

plt.figure(figsize=(16/2.54, 12/2.54))
ax = sns.heatmap(stromemissionen_matrix, cmap=custom_cmap_rot, annot=False, fmt=".2f", yticklabels=range(24), xticklabels=range(1, 366), cbar_kws={'label': 'CO₂-Emissionsfaktor des Strommix [g/kWh]'})

# Customize x-ticks: Show every 50st day
ax.set_xticks(np.arange(0, stromemissionen_matrix.shape[1], 50))
ax.set_xticklabels(np.arange(0, stromemissionen_matrix.shape[1] + 1, 50), fontsize=11, rotation=0)

# Customize y-ticks: Show every third hour
ax.set_yticks(np.arange(0, 24, 3))
ax.set_yticklabels(np.arange(0, 24, 3), fontsize=11)

plt.xlabel('Tag des Jahres', fontsize=12)
plt.ylabel('Stunde des Tages', fontsize=12)
plt.title('Stündlicher CO₂-Emissionsfaktor des Strommix', fontsize=13)

ax.invert_yaxis()

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=11)
cbar.set_label('g/kWh', fontsize=12)

#plt.savefig("StromemissionenHeatmap.pdf", format="pdf")

plt.show()