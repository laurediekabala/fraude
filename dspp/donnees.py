import matplotlib.pyplot as pd
import numpy as pn
import seaborn as sn
import plotly.express as pl
from charger import charge
import streamlit as st

donnees= r"E:\fraude\donnees\bank.csv"
data= charge.chargement(donnees)


           