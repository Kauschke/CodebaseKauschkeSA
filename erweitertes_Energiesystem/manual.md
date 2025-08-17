# Python-Skript für die Optimierung des erweiterten Energiesystems

Das Python-Skript hat die folgende Gliederung, welche als Kommentar im Code wiederzufinden ist:

## 1. Grundlagen
- 1.1 Import der Bibliotheken
- 1.2 standardisierte Farben (Colorcoding)
- 1.3 Einlesen der pickle-Dateien aus dem "Inputs"-Skript

## 2. Modellierung in FINE
- 2.1 Initialisierung des "energy system model"
- 2.2 Anlegen der FINE-Klassen

## 3. Zwischenberechnungen

## 4. Optimierung des Energiesystems
- 4.1 Ausführung der Optimierung
- 4.2 Ausgabe der TAC und THGE
- 4.3 Definition multikritielle Zielfunktion
- 4.4 Festlegung von TAC-/THGE-Obergrenzen
- 4.5 Abspeichern der Simulationsergebnisse in Excel-Sheets

## 5. Datenausgabe
- 5.1 Darstellung der Pareto-Front
- 5.2 Einlesen/Abspeichern der Optimierungsergebnisse
- 5.3 Ausgelegte Leistungen
- 5.4 Ausgabe der Kosten
- 5.5 Ergebnisse Betriebsoptimierung

## 6. Berechnung der KPIs

## 7. Piecharts Energiebedarfe
- 7.1 Pie Chart zur Jahreskälteleistung
- 7.2 Strommengen in kWh
- 7.3 Pie Chart zur Aufteilung TAC

## 8. Heatmaps
- 8.1 Funktionsdefinition
- 8.2 Funktionsaufruf

## 9. Plots für Gewichtungsparameter
- 9.1 Histogram Auslegungen alle weight values
- 9.2 Conversion Kapazitäten über Gewichtungsfaktoren
- 9.3 Storage Kapazitäten über Gewichtungsfaktoren
- 9.4 SourcesSinks Jahresenergie über Gewichtungsfaktoren
- 9.5 Conversion Jahresenergie über Gewichtungsfaktoren
- 9.6 Storage Jahresenergie Discharge über Gewichtungsfaktoren
- 9.7 Heizungswasserspeicher Jahresenergie Discharge
- 9.8 LIB Kältespeicher Jahresenergie Discharge

## 10. Line Plots
- 10.1 Optimierungsergebnisse strukturiert einlesen
- 10.2 Zeitspannen Sommer- und Winterwoche
- 10.3 Winterwoche Strom weight co2
- 10.4 Sommerwoche Strom weight co2
- 10.5 Ladecharakterisierung Speichertechnologien
- 10.6 Winterwoche Raumwärme weight co2
- 10.7 Sommerwoche Raumwärme weight co2
- 10.8 Winterwoche Kälte weight co2
- 10.9 Speicherfüllstände
