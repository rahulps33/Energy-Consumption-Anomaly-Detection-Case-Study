# Energy Tariffs
The distinction between, e.g., `[...] Tarif 0` and `[...] Tarif 1` of the same name is legacy information. There should be no distinction at this point. 

**TODO**: Please check what the behavior in our data is and report it here per building and case: Do both columns have the same values? Does one have all the information and the other one has only zero values?

| Building | repeated columns | other case | 
| --- | --- |  --- | 
| Chemie | BV+Arbeit Tarif 1 and BV+Arbeit tariflos have the same values | WV+Arbeit tariflos is empty while WV+Arbeit Tarif 1 has all the information|  |
| Großtagespflege | 1-CS89 72 01 01_WV-T1  and CS-CS89 72 01 01_WV-tariflos, 1-CS89 72 01 01_BV+T1  and CS-CS89 72 01 01_BV+tariflos, 1-CS89 72 01 03_WV+T1 and CS-CS89 72 01 03_WV+tariflos , 1-CS89 72 01 03_WV-T1 and CS-CS89 72 01 03_WV-tariflos, 1-CS89 72 01 03_BV+T1 and CS-CS89 72 01 03_BV+tariflos, 1-CS89 72 01 04_WV+T1 and CS-CS89 72 01 04_WV+tariflos|   |  |
| OH12 |  WV+Arbeit Tarif 1 has all the information  |   |  |
| EF42 |  1_CN33 71 02 03_WV+tariflos & 1_CN33 71 02 03_WV+T1, 1_CN33 71 02 03_BV+tariflos & 1_CN33 71 02 03_BV+T1, 1_CN33 71 02 03_BV-tariflos & 1_CN33 71 02 03_BV-T1|     | |


# Meaning of the standardized OBIS codes
OBIS codes are used to identfy the type of data that was measured. Here are some general resources to learn about what every digit in the code means in detail: 
* [OBIS codes German/English and detailed descriptions (German)](https://www.kbr.de/de/obis-kennzeichen/obis-kennzeichen#obis-kennzeichensystem) (click on the red German names in the table `OBIS A (Medien)` to get to the detailed descriptions)
* [Description of OBIS code](https://www.promotic.eu/en/pmdoc/Subsystems/Comm/PmDrivers/IEC62056_OBIS.htm) (english, examples, but no detailed explanations)
* [Codeliste der OBIS-Kennzahlen und Medien](https://www.elektronische-vertrauensdienste.de/DE/Beschlusskammern/BK06/BK6_83_Zug_Mess/835_mitteilungen_datenformate/Mitteilung_21/EDIFACT-Konsultaionsdokumente/Codeliste_OBIS-Kennzahlen_Medien_2_4.pdf?__blob=publicationFile&v=1) (German, detailed explanations)

They follow the general structure `A-B:C.D.E*F`:
- The `A` group specifies the medium (0=abstract objects, 1=electricity, 6=heat, 7=gas, 8=water ...)
- The `B` group specifies the channel. Each device with multiple channels generating measurement results, can separate the results into the channels.
- The `C` group specifies the physical value (current, voltage, energy, level, temperature, ...)
- The `D` group specifies the quantity computation result of specific algorythm
- The `E` group specifies the measurement type defined by groups A to D into individual measurements (e.g. switching ranges)
- The `F` group separates the results partly defined by groups A to E. The typical usage is the specification of individual time ranges.

In the following tables, we have collected some information about known codes. These are for a wider collection of buildings than you have worked with so far. 

## Wärme (heat energy, `B=6`)
| OBIS-code | name | unit | building | description |
| --- | --- | --- | --- | --- |
| 6-1:1.8.1 | Wärmeenergie | kWh | EF40, EF40a, EF44, EF50, EF50, EF72, Erich-Brost, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G1-F1, Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, HGÜ, Physik, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1, EF 61+ IBZ, H-Bahn Büro |  |
| 6-1:80.7.1 | Durchfluss | m<sup>3</sup>/h | EF40, EF44, EF72, Erich-Brost, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G1-F1Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, HGÜ, Physik, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1 | **Amount of water** flowing through the pipes of the observed part of the heating system. |
| 6-1:80.8.1 | Volumen | m<sup>3</sup> | EF40, EF72, EF42, E-Technik, OH14, IRF, OH12, Chemietechnik G1-F1Chemietechnik G2-F2, Chemietechnik G3-F3, Delta, Physik, Bibliothek, HG II, Sport |  |
| 6-2:80.8.1 | Volumen Kanal 2 | m<sup>3</sup> | EF40 |  |
| 6-3:80.8.1 | Volumen Kanal 3 | m<sup>3</sup> | EF40 |  |
| 6-1:81.7.1 | Vorlauftemperatur | °C | EF40, EF44, EF72, Erich-Brost, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G1-F1Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, HGÜ, Physik, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1 | Temperature of the heated water before entering the heating pipes and radiators |
| 6-1:82.7.1 | Rücklauftemperatur | °C | EF40, EF44, EF72, Erich-Brost, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G1-F1Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, HGÜ, Physik, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1 | Temperature of the water after running through the heating pipes (at the end of the circulation). This water will then be reheated. |
| 6-1:83.7.1 | Temperaturdifferenz | K | EF40, EF72, E-Technik, OH14, IRF, OH12, Chemietechnik G1-F1Chemietechnik G2-F2, Chemietechnik G3-F3, Bibliothek, HG II, Sport | Temperature difference between _Vorlauftemperatur_ and _Rücklauftemperatur_ |
| 6-1:84.7.1 | Wärmeleistung | kW | EF40, EF44, EF72, Erich-Brost, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G1-F1Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, Physik, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1 | Usable heat energy |


| OBIS-code | Bezeichnung | english name | unit | building | description |
| --- | --- | --- | --- | --- | --- |
| 0-1:0.01.0 | Gerätenummer | device number | \- | EF40, E-Technik, OH14 |  |
| 0-1:0.2.1 | Firmwareversions Nr. | firmware version nr. | \- | EF40, E-Technik, Delta |  |
| 0-1:129.0.1 | Impulswert Kanal 1 | impulse value channel 1 | \- | EF40, E-Technik, Delta |  |
| 0-1:129.0.2 | Impulswert Kanal 2 | impulse value channel 2 | \- | EF40, E-Technik, Delta |  |
| 0-1:129.0.3 | Impulswert Kanal 3 | impulse value channel 3 | \- | EF40, E-Technik, Delta |  |
| 0-1:96.8.0 | Betriebsstunden | operating hours | h | EF40, E-Technik, OH12, Delta, Sport |  |
| 0-1:96.96.0 | Fehlerstunden | hours in error state | h | EF40, E-Technik, OH12, Delta, Sport |  |

## Elektrizität (electricity, `B=1`)
The directions (`+`/`-`) denote the direction of the flow of energy:
* `+`: consumption from the energy grid
* `-`: returning energy to the grid

| OBIS-code | name | unit | building | description |
| --- | --- | --- | --- | --- |
| 1-1:1.25 | P Summe | kW | EF40, EF40a, EF50, Erich-Brost, Mathe, OH12, Chemie, HGÜ, Physik, Bibliothek, HG II, Mensa1 | Last mean value? positive active energy |
| 1-1:1.7.0 | WV+ Momentanwert Tariflos | kW | EF50, Mathe, Chemie | positive active instantaneous power |
| 1-1:1.8.0 | WV+ Arbeit tariflos | kWh | EF40, EF40a, EF44, EF50, Erich-Brost, Mathe, EF42, OH12, Chemietechnik G2-F2, Chemietechnik G3-F3, HGÜ, Physik, Bibliothek, Mensa1 | positive active energy total |
| 1-1:1.8.1 | WV+ Arbeit Tarif 1 | kWh | EF40, EF40a, EF44, EF50, Erich-Brost, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, HGÜ, Physik, Audimax, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1, EF 61+ IBZ, H-Bahn Büro | positive active energy total tariff 1 |
| 1-1:1.8.2 | WV+ Arbeit Tarif 2 | kWh | HGÜ | positive active energy total tariff 2 |
| 1-1:2.8.0 | WV- Arbeit tariflos | kWh | EF50, Erich-Brost, Mathe, Chemie, Chemietechnik G2-F2, Bibliothek, Mensa1 | negative active energy total |
| 1-1:2.8.1 | WV- Arbeit Tarif 1 | kWh | EF50, Erich-Brost, Mathe, Chemie, Chemietechnik G2-F2, Bibliothek, HG II, Mensa1 | negative active energy tariff 1 |
| 1-1:3.8.0 | BV+ Arbeit tariflos | kvarh | EF40, EF40a, EF44, EF50, Erich-Brost, Mathe, EF42, Chemie, Chemietechnik G2-F2, Physik, Bibliothek, Mensa1 | positive reactive energy total |
| 1-1:3.8.1 | BV+ Arbeit Tarif 1 | kvarh | EF40, EF40a, EF44, EF50, Erich-Brost, Mathe, EF42, Chemie, Chemietechnik G2-F2, Physik, Bibliothek, Mensa1 | positive reactive energy tariff 1 |
| 1-1:4.8.0 | BV- Arbeit tariflos | kvarh | EF40, EF40a, EF44, EF50, Erich-Brost, Mathe, EF42, Chemie, Chemietechnik G2-F2, Physik, Bibliothek, Mensa1 | negative reactive energy total |
| 1-1:4.8.1 | BV- Arbeit Tarif 1 | kvarh | EF40, EF40a, EF44, EF50, Erich-Brost, Mathe, EF42, Chemie,  Chemietechnik G2-F2, Physik, Bibliothek, Mensa1 | negative reactive energy tariff 1 |


| OBIS-code | name | unit | building | description |
| --- | --- | --- | --- | --- |
| 0-1:97.128.0 | error flags | \- | EF50, Mathe, E-Technik, OH12, Chemie, Mensa1 |  |


## Wasser (water, `B=8`)
| OBIS-code | name | unit | building | description |
| --- | --- | --- | --- | --- |
| 8-1:80.8.1 | Volumen Kanal 1 | m<sup>3</sup> | EF40, EF40a, EF44, EF50, Mathe, EF42, E-Technik, OH14, IRF, OH12, Chemie, Chemietechnik G1-F1, Chemietechnik G2-F2, Chemietechnik G3-F3, CP OH4a, Delta, Frauenhofer, HGÜ, Physik, Bibliothek, HG II, Kita Hokido, Mensa1, Sport, SRG1, EF 61+ IBZ, H-Bahn Büro | cold water |
| 8-1:80.7.1 | Durchfluss | m³/h | Physik | cold water

| OBIS-code | name | unit | building | description |
| --- | --- | --- | --- | --- |
| 8-1:0.01.0 | Fabrikationsnummer | - | EF40a, Physik, H-Bahn Büro | fabrication number


## Gas (gas, `B=7`)
| OBIS-code | name | unit | building | description |
| --- | --- | --- | --- | --- |
| 7-1:80.8.1 | Volumen Eingang DE1 | m<sup>3</sup> | CP OH4a, Mensa1, DEW-Gasübergabe |


# Meaning and structure of internal codes
The facility management have their own internal codes. Take a water meter: `CN 13 41 xx xx`
* `CN` or `CS` denotes campus north and campus south
* `13` is the internal building number, here: `EF 50`. Check the `Gebäudeliste_Gebäudeplan` uploaded to Sciebo (contains a map as well) in the folder [documents](https://tu-dortmund.sciebo.de/apps/files/?dir=/case%20study%20energy/students/documents&fileid=300229309)!
* `41` is the internal code for a work unit, here `water`. Check the `Gewerkeplan` uploaded to Sciebo in the folder [documents](https://tu-dortmund.sciebo.de/apps/files/?dir=/case%20study%20energy/students/documents&fileid=300229309)!
* the last two numbers are used to generate a unique identifier

There can be multiple meters of the same type per building, e.g., one main meter for the whole building and several sub-meters (could be for a floor or a particular part of the building, e.g., a café that pays for it's own consumption). We do not know the exact use of the meters and how they relate to each other so it's best not to make strong assumptions.

## Unique, friendly column names 
| name | abbreviation |
| --- | --- | 
| Wärmeenergie Tarif 1 | WTarif1 | | 
| Wärmeenergie total | Wtotal | | 
| Durchfluss | Dfluss | |
| Volumen | Vol | |
| Volumen Kanal 1 | VK1 | | 
| Rücklauftemperatur | Rücklauftmp | | 
| Vorlauftemperatur | Vorlauftmp | | 
| Temperaturdifferenz | TmpDiff | | 
| Wärmeleistung | Wleistung | | 
| WV+ Arbeit Tarif 1 | WV+T1 | | 
| WV- Arbeit Tarif 1 | WV-T1 | | 
| WV+ Arbeit tariflos | WV+tariflos || 
| WV- Arbeit tariflos| WV-tariflos || 
| BV+ Arbeit Tarif 1 | BV+T1  | | 
| BV- Arbeit Tarif 1 | BV-T1 || 
| BV+ Arbeit tariflos | BV+tariflos || 
| BV- Arbeit tariflos| BV-tariflos || 
| Betriebsstunden | Betriebsstd | | 
| Fehlerstunden | Fehlerstd | | 
| P Summe | PSum | | 
| WV+ Momentanwert Tariflos | WV+Momtrflos ||
| Fehler Flags | FehFlag ||


## Used format: 
First digit of OBIS Kennzahl_all digits of Beschreibung_abbreviation

Example for Wärmeenergie total of OH12: `6_CN45 11 01 01_Wtotal` and `6_CN45 21 01 01_Wtotal`

