import matplotlib.pyplot as pd
import numpy as pn
import seaborn as sn
import plotly.express as pl
from charger import charge
import streamlit as st
from pathlib import Path
donnees= Path(__file__).parent / "donnees" / "bank.csv"
data= charge.chargement(donnees)


           