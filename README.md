# Kandidatarbete – Identifiering av förvärvbara YouTube-barnkanaler

Ett maskininlärningsprojekt som analyserar europeiska YouTube-kanaler för barn för att identifiera kanaler med hög förvärvspotential. Projektet klassificerar kanaler i två kategorier: **Rising Stars** (kanaler på uppgång) och **Sleeping Giants** (undervärderade kanaler med stark engagemangsprofil).

---

## Innehållsförteckning

- [Projektöversikt](#projektöversikt)
- [Förutsättningar](#förutsättningar)
- [Installation](#installation)
- [Konfiguration](#konfiguration)
- [Hur man kör projektet](#hur-man-kör-projektet)
- [Projektstruktur](#projektstruktur)
- [Modulbeskrivningar](#modulbeskrivningar)
- [Dataflöde](#dataflöde)
- [Modeller](#modeller)
- [Output](#output)

---

## Projektöversikt

Projektet hämtar data från YouTube Data API v3 för barnkanaler i Europa, bygger features, tränar maskininlärningsmodeller och rankar kanaler efter förvärvspotential.

**Två typer av kanaler identifieras:**

| Typ | Beskrivning |
|-----|-------------|
| **Rising Star** | Kanal med starkt momentum – hög prenumeranttillväxt de kommande 6 månaderna |
| **Sleeping Giant** | Undervärderad kanal – stark engagemangsprofil men för få prenumeranter relativt sin aktivitet |

---

## Förutsättningar

- Python 3.10+
- En **YouTube Data API v3-nyckel** (gratis via Google Cloud Console)
- Git

---

## Installation

1. Klona repot:
```bash
git clone git@github.com:idaeskilsson/Kandidatarbete.git
cd Kandidatarbete
```

2. Skapa och aktivera en virtuell miljö:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

3. Installera beroenden:
```bash
pip install -r requirements.txt
```

---

## Konfiguration

Skapa en fil som heter `.env` i projektets rotkatalog med din YouTube API-nyckel:

```
YOUTUBE_API_KEY=din_api_nyckel_här
```

Du kan skaffa en API-nyckel på [Google Cloud Console](https://console.cloud.google.com/):
1. Skapa ett projekt
2. Aktivera **YouTube Data API v3**
3. Skapa en API-nyckel under Credentials

> **OBS:** Gratis-kvoten är 10 000 enheter/dag. Datainsamlingen förbrukar en stor del av denna kvot. Om kvoten tar slut återställs den vid 07:00 svensk tid (midnatt Pacific Time).

---

## Hur man kör projektet

### Alternativ 1 – Med befintlig data
Om du redan har datafiler i `data/raw/` (från en tidigare körning):
```bash
python main.py
```

### Alternativ 2 – Hämta ny data från YouTube API
```bash
python main.py --fetch
```

Detta hämtar kanaldata och videor från YouTube innan modellerna tränas.

### Vad händer när du kör?

1. **(Valfritt)** Datainsamling från YouTube API
2. Preprocessing – filtrering av kanaler
3. Bygg target-variabel (YouTube-värde per video)
4. Bygg features (engagemang, tillväxt, etc.)
5. Kör Rising Star-modellen
6. Kör Sleeping Giant-modellen

---

## Projektstruktur

```
Kandidatarbete/
├── main.py                    # Huvudingångspunkt – kör hela pipelinen
├── requirements.txt           # Python-beroenden
├── .env                       # API-nyckel (skapas manuellt, pushas INTE till GitHub)
├── .gitignore
│
├── src/
│   ├── youtube_client.py      # Skapar YouTube API-klient via API-nyckel
│   ├── collect_data.py        # Hämtar kanaler och videor från YouTube API
│   ├── preprocess.py          # Rensar och filtrerar rådata
│   ├── build_target.py        # Beräknar YouTube-värde (yv) per kanal
│   ├── build_features.py      # Bygger ML-features från kanaldata
│   ├── filter_channels.py     # Filtrerar bort storbolag och kanaler utan tydlig IP
│   ├── rising_star_model.py   # XGBoost-modell för tillväxtprediktering
│   └── sleeping_giant_model.py# XGBoost + K-Means för undervärderade kanaler
│
├── data/                      # Genereras automatiskt (pushas INTE till GitHub)
│   ├── raw/                   # Rådata direkt från API
│   ├── interim/               # Mellansteg i pipelinen
│   └── processed/             # Färdig data för modellträning
│
└── outputs/                   # Resultat och grafer (pushas INTE till GitHub)
    └── predictions/           # CSV-rankingar för Rising Stars och Sleeping Giants
```

---

## Modulbeskrivningar

### `youtube_client.py`
Skapar en autentiserad YouTube API-klient. Läser `YOUTUBE_API_KEY` från `.env`-filen och returnerar ett `googleapiclient`-objekt som används av `collect_data.py`.

---

### `collect_data.py`
Hämtar data från YouTube API i sju steg:

| Steg | Beskrivning |
|------|-------------|
| 1 | Söker kanaler direkt via sökfrågor på 14 språk/regioner |
| 2 | Kategoribaserad sökning – söker videor i kategorierna Entertainment (24) och Education (27) och extraherar kanalerna bakom dem |
| 3 | Hämtar kanaldetaljer (prenumeranter, videor, visningar, land) för alla hittade kanaler |
| 4 | Hämtar *featured channels* – kanaler som de redan hittade kanalerna rekommenderar |
| 5 | Seed-expansion – söker liknande kanaler baserat på de 20 mest populära seedkanalerna |
| 6 | Hämtar kanaldetaljer för alla nya kanaler från steg 4–5 |
| 7 | Hämtar upp till 60 videor per kanal (visningar, likes, kommentarer, längd) |

**Sparar till:**
- `data/raw/channels_raw.csv` – kanaldata
- `data/raw/videos_raw.csv` – videodata
- `data/raw/search_results_raw.csv` – råa sökresultat
- `data/interim/` – checkpoints under körningens gång

---

### `preprocess.py`
Rensar och filtrerar rådata:
- Tar bort dubbletter
- Konverterar datum till korrekt format
- **Behåller bara barnkanaler** (`madeForKids = True` eller okänt)
- **Behåller bara europeiska kanaler** (26 länder: SE, GB, DE, FR, PL, ES, IT, NL, BE, AT, FI, PT, IE, CH, CZ, HU, RO, GR, SK, HR, BG, LT, LV, EE, NO, DK)

**Sparar till:** `data/interim/channels_clean.csv`, `data/interim/videos_clean.csv`

---

### `build_target.py`
Beräknar targetvariabeln **yv** (YouTube-värde per video):

```
yv = avg_views_per_video × monetization_rate × CPM / 1000
```

- `monetization_rate` = 0.45 (45 % av visningar är monetiserade, branschstandard)
- `CPM` = land-specifikt värde (t.ex. GB = 4.5, SE = 3.5, DE = 4.0, standardvärde = 2.5)

**Sparar till:** `data/processed/model_dataset_with_target.csv`

---

### `filter_channels.py`
Kör två filter innan ML-modellerna tränas:

**Filter 1 – Förvaltningsmöjlighet:**
Tar bort kanaler som ägs av stora etablerade medieaktörer (Disney, BBC, Nickelodeon, SVT, NRK, Cocomelon m.fl.) eller har fler än 10 miljoner prenumeranter. Dessa är inte förvärvbara.

**Filter 2 – Tydlig IP:**
Behåller bara kanaler med återkommande karaktärer eller värld. Mäts genom att se hur ofta samma innehållsord (exkl. stopord) förekommer i videotitlarna. Om det vanligaste karaktärsordet dyker upp i minst 25 % av titlarna räknas kanalen som att ha tydlig IP.

---

### `build_features.py`
Bygger ML-features från den filtrerade kanaldata:

| Feature | Beskrivning |
|---------|-------------|
| `channel_age_days` | Antal dagar sedan kanalstart |
| `subscribers_per_video` | Prenumeranter dividerat med antal videor |
| `views_per_video_reported` | Totala visningar dividerat med antal videor |
| `upload_frequency` | Videor per dag sedan kanalstart |
| `engagement_rate_like` | Genomsnittliga likes per video / genomsnittliga visningar per video |
| `engagement_rate_comment` | Genomsnittliga kommentarer per video / genomsnittliga visningar per video |
| `is_made_for_kids` | Binär flagga för barninnehåll |

**Sparar till:** `data/processed/model_dataset.csv`

---

### `rising_star_model.py`
Identifierar kanaler med hög prenumeranttillväxt de kommande 6 månaderna.

**Kräver:** `data/raw/channel_history.csv` med månadsvis historikdata per kanal i formatet:
```
channel_id | month | subscribers | total_views | video_count | likes | comments | channel_age_days | country
```

**Metod:**
1. Beräknar momentum-features för 6-månadersfönster: tillväxttakt, acceleration, engagemangstrend, uppladdningskonsistens, viral ratio
2. Tränar en **XGBoost-klassificerare** som predictar om en kanal tillhör topp 25 % i tillväxt inom sin storleksgrupp (liten: 1k–10k, medel: 10k–50k, stor: 50k+)
3. Genererar SHAP-diagram för att förklara vilka features som driver prediktionerna
4. Rankar kanaler efter sannolikhet att tillhöra topp 25 %

**Sparar till:**
- `outputs/predictions/rising_star_1k.csv` – kanaler med 1 000+ prenumeranter
- `outputs/predictions/rising_star_10k.csv` – kanaler med 10 000+ prenumeranter
- `outputs/predictions/rising_star_50k.csv` – kanaler med 50 000+ prenumeranter
- `outputs/rising_star_shap_bar.png` – SHAP bar-plot
- `outputs/rising_star_shap_beeswarm.png` – SHAP beeswarm-plot

---

### `sleeping_giant_model.py`
Identifierar undervärderade kanaler med stark engagemangsprofil men för få prenumeranter relativt sin aktivitet. Använder två kompletterande metoder:

**Metod 1 – ActivityBenchmark (XGBoost residualanalys):**
- Tränas enbart på välpresterande kanaler (topp 30 % i prenumeranter per storlekssegment)
- Predictar förväntad log(prenumeranter) baserat på aktivitetsprofil (visningar, likes, kommentarer, CPM, videolängd)
- Kanaler med **negativ residual** = har färre prenumeranter än sin aktivitet motiverar = Sleeping Giant

**Metod 2 – EngagementCluster (K-Means klustring):**
- Klustring baserat på engagemangsprofil (like-rate, comment-rate, söksynlighet, uppladdningstakt)
- Körs separat per storlekssegment (mikro: 5k–25k, liten: 25k–100k, medel: 100k–500k, stor: 500k+)
- Klustret med starkast engagemangsprofil identifieras som Sleeping Giants

**Sparar till:**
- `outputs/predictions/sleeping_giant_activity_benchmark.csv`
- `outputs/predictions/sleeping_giant_engagement_cluster.csv`
- SHAP-diagram och klustervisualisering i `outputs/`

---

## Dataflöde

```
YouTube API
    |
    v
collect_data.py  -->  data/raw/channels_raw.csv
                      data/raw/videos_raw.csv
    |
    v
preprocess.py    -->  data/interim/channels_clean.csv
                      data/interim/videos_clean.csv
    |
    v
build_target.py  -->  data/processed/model_dataset_with_target.csv
    |
    v
build_features.py + filter_channels.py
                 -->  data/processed/model_dataset.csv
    |
    +---> rising_star_model.py   -->  outputs/predictions/rising_star_*.csv
    |
    +---> sleeping_giant_model.py --> outputs/predictions/sleeping_giant_*.csv
```

---

## Output

När pipelinen körts klart hittar du resultat i `outputs/predictions/`:

| Fil | Innehåll |
|-----|----------|
| `rising_star_1k.csv` | Rising Stars med 1 000+ prenumeranter, rankade efter tillväxtpotential |
| `rising_star_10k.csv` | Rising Stars med 10 000+ prenumeranter |
| `rising_star_50k.csv` | Rising Stars med 50 000+ prenumeranter |
| `sleeping_giant_activity_benchmark.csv` | Undervärderade kanaler via residualanalys |
| `sleeping_giant_engagement_cluster.csv` | Undervärderade kanaler via klustring |

Varje fil innehåller `channel_id`, `channel_title`, prenumeranttal, engagemangsmått och en rankingpoäng.
