# Data Dictionary: 2025 Crash Data

Source: `2025CrashData.xlsx`, sheet `2025`. It has **6,776 rows** (one per crash) and **137 columns**. Each crash is in calendar year 2025 (Jan 1 to Dec 31).

This dictionary covers the 25 variables most relevant to our questions: driver condition, external conditions, crash intensity, and high-frequency intersections. For a coded field, the spreadsheet holds a numeric **code** column and a text **description** column next to it. Both are listed here; group and filter on the code column.

Fields for vehicles 2 and 3 (and drivers 2 to 6) follow the same structure as the vehicle 1 / driver 1 fields below.

## Variables

| # | Col | Variable (exact header) | Type | Description | Notes |
|---|-----|-------------------------|------|-------------|-------|
| **Location** |||||
| 1 | AY | `LOCKEY` | int ID | Unique ID for an intersection or a midblock segment. Group by this field to count crashes per location. | 2,992 distinct locations. |
| 2 | H, I, J | `On Street (Name)`, `At Street`, `Between Street (if midblock)` | text | Street names. If `Between Street` is blank, the location is the intersection of *On* and *At*. Otherwise it is a segment of *On* between *At* and *Between*. | 2,480 crashes (37%) are midblock. |
| 3 | DD, DE | `Latitude`, `Longitude` | decimal | Crash coordinates, used for maps. | About 2% are blank. Two `Latitude` cells hold case numbers (`C5-…`), so convert the column to numeric and coerce errors. |
| **Time** |||||
| 4 | B | `LPD Date` | date | Date of the crash. | `LTU Accident Date` (AD) is nearly identical; they differ on 5 rows. |
| 5 | C | `Day of Week 1=Sun` | int 1–7 | 1 = Sunday … 7 = Saturday. | |
| 6 | D | `Time of Accident` | int HHMM | 24-hour clock stored as a number. For example, `13` = 00:13 and `1752` = 17:52. | **`9999` = unknown (552 rows).** Exclude these before computing hours. Per the City, these are likely hit-and-runs or late reports where the driver could not recall the time. |
| **Driver condition** |||||
| 7 | AS / AT | `Driver 1 Condition Code` / `Driver 1 Condition` | coded | Physical or mental state of driver 1, as observed by the responding officer. | See [code list](#driver-1-condition) and [City clarifications](#city-clarifications) for how to treat blanks and combine drivers. |
| 8 | Z | `Driver 1 Drinking or BAC` | mixed | A blood alcohol value (e.g. `0.179`), `Y` (drinking, no BAC recorded), `N`, or `99` (unknown). | Blank in 96% of rows. 302 rows are non-blank; 118 of them are `Y`. |
| 9 | CR / CS | `Driver 1 Distraction Code` / `Desc` | coded | 0 Not distracted · 1 Talking/listening · 2 Manually operating (phone/game) · 3 Other action (looking away, etc.) · 99 Unknown | 30% are Unknown. |
| 10 | BU / BV | `Driver 1 Contributing Circumstances Code` / text | coded | Driver action that contributed to the crash. | See [code list](#driver-1-contributing-circumstances). |
| **External conditions** |||||
| 11 | AG / AH | `Weather Condition Code` / `Weather Condition` | coded | Weather at the time of the crash. | See [code list](#weather-road-and-light). |
| 12 | AE / AF | `Road Condition Code` / `Road Condition` | coded | Road surface condition. | See [code list](#weather-road-and-light). |
| 13 | AI / AJ | `Light Condition Code` / `Light Condition` | coded | Daylight or darkness, and street lighting. | See [code list](#weather-road-and-light). |
| 14 | BS / BT | `Road Contributing Circumstances Code` / text | coded | Roadway factor that contributed, such as glare, an animal, a work zone or the road surface. | See [code list](#road-contributing-circumstances). |
| 15 | N | `Ped Involved` | flag | `X` = a pedestrian was involved. Blank = none. | 99 crashes. |
| 16 | O | `Animal Involved` | flag | `X` = an animal was involved. Blank = none. | 34 crashes. |
| 17 | BM / BN | `Fixed Object Hit Code` / `Fixed Object Hit.1` | coded | The object struck, such as a pole, sign or tree. 0 = none. | See [code list](#fixed-object-hit). The flag column `Fixed Object Hit` (P) uses `C`/`X`, which is undocumented, so use the code column instead. |
| 18 | EF | `Type of NMV Involved` | text | The non-motor-vehicle or vulnerable road user involved: `Ped`, `Bike`, `MC` (motorcycle), `Scooter`, `ATV`, etc. | Blank in 96% of rows. |
| **Crash type & intensity** |||||
| 19 | E / F | `LPD Incident Code Number` / `Abrev.` | coded | Police classification of severity and hit-and-run. | See [code list](#lpd-incident-code). |
| 20 | Q / R | `Type of Accident Code` / `Type of Accident` | coded | Collision type. | See [code list](#type-of-accident). |
| 21 | G | `Number of Involved Vehicles` | int | Count of vehicles involved. | Ranges from 1 to 6; 80% of crashes involve 2. |
| 22 | DF–DI | `# of Fatalities (1)`, `# of Suspected Serious Injuries (2)`, `# of Suspected Minor Injuries (3)`, `# of Possible Injuries (4)` | int | Number of people at each severity level, from (1) most severe to (4) least. | Totals: 18 fatalities, 93 serious, 1,301 minor, 913 possible. |
| 23 | S | `Number of Injuries` | int | Total injured. | **`99` = unknown (19 rows).** Prefer the columns in row 22 for totals. |
| **Location characteristics** |||||
| 24 | CP / CQ | `Intersection Control Code` / `Desc` | coded | 1 No control · 2 Signalized · 3 Stop, all-way · 4 Stop, partial · 5 Yield · 97 Not applicable (midblock) | |
| 25 | CA / CB, CE | `Road Classification Code` / `Type`; `ADT` | coded; int | Class of the road, or of the pair of crossing roads at an intersection. `ADT` = Average Daily Traffic (vehicles per day). | Codes: 0 Local · 1 Major · 2 Collector · 4 Expressway · 11 Major/Major · 12 Major/Collector · 13 Major/Local · 22 Collector/Collector · 23 Collector/Local · 33 Local/Local. The description is blank in 489 rows even though the code is present. Use ADT to turn crash counts into crash *rates* when comparing intersections. |

## Code lists

### Driver 1 Condition
0 (blank) · 1 Apparently normal · 2 Asleep or fatigued · 3 Emotional · 4 Illness (sick or fainted) · 5 Physical impairment · 6 Medication/drugs/alcohol · 97 Not applicable · 98 Other · 99 Unknown

### Driver 1 Contributing Circumstances
0 No contributing action · 1 Disregarded red light · 2 Disregarded stop sign · 3 Disregarded road markings · 4 Disregarded traffic sign · 5 Failure to keep in proper lane · 6 Failed to yield right of way · 7 Followed too closely · 8 Improper backing · 9 Improper passing · 10 Made improper turn · 11 Operating vehicle carelessly · 12 Reckless/aggressive · 13 Over-correcting/steering · 14 Ran off roadway · 15 Swerved to avoid hazard · 16 Wrong side/wrong way · 98 Other · 99 Unknown

### Weather, Road and Light
- **Weather:** 1 Blowing sand/soil/dirt · 2 Blowing snow · 3 Clear · 4 Cloudy · 5 Fog/smog/smoke · 6 Freezing rain/drizzle · 7 Rain · 8 Severe crosswinds · 9 Sleet or hail · 10 Snow · 98 Other · 99 Unknown
- **Road:** 1 Dry · 2 Ice/frost · 3 Mud/dirt/gravel · 5 Sand · 6 Slush · 7 Snow · 8 Water · 9 Wet · 98 Other · 99 Unknown
- **Light:** 1 Daylight · 2 Dawn/dusk · 3 Dark, lighted roadway · 4 Dark, not lighted · 5 Dark, unknown lighting · 98 Other · 99 Unknown

### Road Contributing Circumstances
0 None · 1 Absence of sidewalks · 2 Animal(s) · 3 Prior crash · 4 Prior non-recurring incident · 5 Backup due to regular congestion · 6 Debris · 7 Glare · 8 Obstructed sidewalks · 10 Obstruction in roadway · 11 Related to bus stop · 12 Road surface condition · 13 Roadway width restriction · 15 Shoulders · 17 Traffic control device inoperative · 18 Traffic incident · 19 Vision obstruction · 20 Weather conditions · 21 Work zone · 98 Other · 99 Unknown

### Fixed Object Hit
0 None · 1 Street light pole · 2 Traffic signal pole · 3 Signal control box · 4 Sign · 6 Mailbox · 7 Guard rail · 8 Object (private property) · 9 Bridge structure/rail · 10 Tree · 11 Utility pole · 13 Fire hydrant · 14 Signal head · 15 RR crossing arm · 17 Parking meter · 18 Barricade · 19 Signal cable · 20 Utility box · 21 Tubular marker · 22 Guy wire · 99 Other

### LPD Incident Code
3000 `ACC` general accident · 3021 `ACC H&R` hit and run · 3111 `ACC PD` property damage · 3121 `ACC PD H&R` · 3211 `ACC INJ` injury · 3221 `ACC INJ HR` · 3311 `ACC FAT` fatal

Six rows have stray codes (`0`, `3011`, `30111`).

### Type of Accident
1 Right angle · 2 Turning · 3 Rear end · 4 Side swipe · 5 Parked · 6 Backing · 7 Head on · 8 Overhead · 9 Driveway · 10 Other · 11 Ran off road · 12 Unknown

## City clarifications

Answers from City of Lincoln staff to our questions about the data (email reply, Nov 2026).

- **Vehicle type (`Type of Veh 1/2/3`, columns K/L/M):** `TRUCK` can mean a pickup or a semi; the data does not distinguish them.
- **`Time of Accident` = 9999:** likely a hit-and-run with unknown time, or a late report where the driver did not recall the time.
- **Driver condition** is the responding officer's observation and discretion. `UNKNOWN` is mostly used for at-fault drivers in hit-and-runs; `NOT APPLICABLE` is used when no driver would be present (e.g. a parked vehicle). The two may sometimes be used interchangeably.
- **Blank driver condition:** if a vehicle is listed in K/L/M but that driver's condition is blank, treat it as `UNKNOWN`.
- **`MEDICATION/DRUGS/ALCOHOL`** can mean suspected or confirmed, so call it *potentially impaired*.
- **Condition vs. Distraction vs. Contributing Circumstances** are separate measures, and one crash can appear in all three. Condition is the driver's physical appearance. Distraction is whether something such as a phone took attention from the road. Contributing Circumstances is whether road conditions or driver behavior contributed.
- **Driver order (1, 2, 3)** does not tell you who was at fault.
- **No individual driver injury severity** is available. KSI-by-condition counts *crashes*, not injured drivers.
- **One condition per crash.** To assign a single driver condition to a crash:
  1. Use driver 1's condition unless it is Normal or Not Applicable.
  2. Otherwise, use the first listed driver condition that is not Normal or Not Applicable.
  3. Use Normal only when every driver is Normal or Not Applicable.

  This is implemented in the driver-condition chunk of `analysis/Crash_Data_R.Rmd`.

## Cleaning reminders
- **Placeholder values:** `Time of Accident` = 9999, `Number of Injuries` = 99, and driver `Age` = 0 (1,005 rows for driver 1) all mean *unknown*. Any coded field uses 99 = Unknown, 98 = Other and 97 = Not applicable.
- **`CASE` is empty** because it was removed for confidentiality. Fill it with 1…6776 in the original row order to give each crash a reference ID (per `NOTES_ON_CRASH_DATA.txt`).
- **Unknown values:** for driver condition, distraction and contributing circumstances, decide whether to drop Unknown or report it as its own category. It is 15–30% of rows in those fields.
