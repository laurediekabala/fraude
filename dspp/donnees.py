import matplotlib.pyplot as pd
import numpy as pn
import seaborn as sn
import plotly.express as pl
from charger import charge
import streamlit as st
import os
# --- Utiliser un chemin relatif vers le fichier CSV ---

# Obtenir le chemin absolu du dossier dspp/
BASE_DIR = os.path.dirname(__file__)

# Construire le chemin vers donnees/bank.csv
donnees = os.path.join(BASE_DIR, "..", "donnees", "bank.csv")

# Normaliser le chemin pour éviter les problèmes Windows/Linux
donnees = os.path.abspath(donnees)
data= charge.chargement(donnees)


           