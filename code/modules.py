##  Importing ##

import os
from bs4 import BeautifulSoup 
from selenium import webdriver
from scipy import stats
from scipy.stats import gaussian_kde
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import pandas as pd 
import numpy as np 
import pickle
import glob

from dotenv import load_dotenv
from decouple import config

import time
from tqdm import tqdm
from random import randint
import datetime
import sys
import string
import calendar

from requests import get  # to make GET request
import requests
from urllib.request import urlopen
from sodapy import Socrata
import json  
import re           
from sklearn.feature_extraction.text import CountVectorizer 
from joblib import Parallel, delayed

import seaborn as sns
#%matplotlib inline # add line to notebooks
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import matplotlib.ticker as ticker
from matplotlib.pyplot import hist
import matplotlib_venn as mv
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
plt.style.use('seaborn-whitegrid')
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime


import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve,recall_score,f1_score,precision_score
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance



from bokeh.models import ColumnDataSource, FactorRange,Legend
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import Spectral6,Turbo256,Viridis,Category20
from bokeh.transform import factor_cmap
from bokeh.io import output_notebook

import folium
from folium.plugins import HeatMap

import ipywidgets as widgets
from IPython.display import display

import imageio
from IPython.display import Image


from global_land_mask import globe

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

"""
## Tokens ##
# rapidApi keys
RAPIDAPI_KEY= os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST= os.environ.get("RAPIDAPI_HOST")
headers = {
      'x-rapidapi-key': RAPIDAPI_KEY,
      'x-rapidapi-host': RAPIDAPI_HOST,
      }
print (headers)
# Socrata keys 
myemail= os.environ.get("myemail")
mypsw= os.environ.get("mypsw")
SocrataToken = os.environ.get("SocrataToken")
print (SocrataToken)

print ("Tokens imported")
"""
print (f"Import succesfull")