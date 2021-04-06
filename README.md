# DataAnalysisExample
SF case study to exercise Data Analysis Tools 

<img src="https://github.com/arita89/DataAnalysisExample/blob/main/figures-Images2GIF/crimeMonthYear/assault/assault.gif" width="600"> |
<img src="https://github.com/arita89/DataAnalysisExample/blob/main/figures-Images2GIF/crimeTime/assault/assault%20in%20time%20.gif" width="300"> |  <img src="https://github.com/arita89/DataAnalysisExample/blob/main/figures/assault_evol.gif" width="300">


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
        ├── Intro & Data preprocessing                          
        ├── Exploratory Analysis                        
        └── ML Correlations and Predictions          
    
```

