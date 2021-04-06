# Analysis of SF data: crime,weather,airbnb
SF case study to exercise Data Analysis Tools.
Note that some plots (folium maps for example) might not be visible without re-running the notebooks, which require the environment built by requirement.txt and to load the datasets.

## Prerequisites

See requirements.txt

To create an env:
> $ conda create --name <EnvName> --file requirements.txt

To load on a pre-existing env:
> $ python -m pip install -r requirements.txt

## Files Description

```
    .
    ├──./code/
        ├── scrapAirbnb.py              # contains .py scripts for data fetching                   
        ├── scrapSocrata.py             # contains .py scripts for data fetching
        └── functions_files.py          # several functions scripts
    ├──../figures/                      # various images outputs
    ├──../figures-Images2GIF/           # various gif outputs
    ├──../data/                         # raw collected data (.csv,.pkl in gitignore)
    └── ../explainerNoteboooks/
        ├── (Gif-creator)
        ├── Intro & Data preprocessing                          
        ├── Exploratory Analysis                        
        └── ML Correlations and Predictions          
    
```

<img src="https://github.com/arita89/DataAnalysisExample/blob/main/figures-Images2GIF/crimeMonthYear/assault/assault.gif" width="800"> |
<img src="https://github.com/arita89/DataAnalysisExample/blob/main/figures-Images2GIF/crimeTime/assault/assault%20in%20time%20.gif" width="400"> |  <img src="https://github.com/arita89/DataAnalysisExample/blob/main/figures/assault_evol.gif" width="400">

