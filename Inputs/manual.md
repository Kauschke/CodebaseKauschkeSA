# Python-Skript für das Einlesen der Inputdaten (Excel/CSV)

Das Python-Skript hat die folgende Gliederung, welche als Kommentar im Code wiederzufinden ist:

## 1. Grundlagen
- 1.1 Import der Bibliotheken
- 1.2 standardisierte Farben

## 2. Einlesen der Daten
- 2.1 Grünenthal-Daten als Excel einlesen
- 2.2  csv mit Strompreisen und -emissionen einlesen (Quelle: Agora Energiewende)

## 3 Energiebedarfe separat abspeichern
- 3.1 Umverteilung der 1 MW Zählerwerte auf vorherige 0er Einträge
- 3.2 Zuweisung der Spalten im Excel-Sheet zu den Energieträgern

## 4. Datenaufbereitung (Ausreißer aus Datenreihe entfernen)
- 4.1 process_series für Extremwerte
- 4.2 Ausreißeruntersuchung, qualitativ
- 4.3 process_outliers für Ausreißer
- 4.4 process_outliers für Zeitreihen mit vielen Nullwerten
- 4.5 Anwendung der Ausreißerkorrektur

## 5 Darstellung der Energiebedarfe
- 5.1 Energiebedarfe für 2024 plotten
- 5.2 Modellierung Strompreise
- 5.3 Modellierung der Preisschwankung +- 20 % des Strompreises
- 5.4 Modellierung des Strompreises als Projektion des Strompreisszenarios 2030
- 5.5 Modellierung des Gaspreises

## 6. Zählerstände
- 6.1  Darstellung Zählerstände 2024
- 6.2 Untersuchung BHKW
 
## 7 Modellierung der PV-Leistung
- 7.1 Datenaufbereitung der PV-Leistung
- 7.2 Plotten der PV-Leistung
- 7.3 Preis für PV-Contracting

## 8 Modellierung der Wärmepumpe und Kompressionskältemaschine (KKM)
- 8.1  Modellierung des COP in Abhängigkeit der Außentemperatur
- 8.2 Versuch der Modellierung des Zusammenhangs zwischen PV-Bereitstellung und WP-Nutzung

## 9 Visualisierung
- 9.1 Datenaufbereitung Energiebedarfe
- 9.2 Violinplot der Energiebedarfe
- 9.3 Violinplot des Strom- und Gaspreises
- 9.4 Heatmap des Strom- und Gaspreises
- 9.5 Violinplot für Strom- und Gasemissionen
- 9.6 Violinplot für Strom- und Gasemissionen

