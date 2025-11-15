import matplotlib.pyplot as plt
import numpy as pn
import seaborn as sn
import plotly.express as pl
from charger import charge
import streamlit as st
import pandas as pd
donnees= r"E:\fraude\donnees\bank.csv"
def univ() :
    data= charge.univarie(donnees)    
def cat_cat() :
    data= charge.cat_cat(donnees)     
def balance() :
    balance= pd.Series
    box=charge.ints(donnees,"balance") 
def age() :
    balance= pd.Series
    box=charge.ints(donnees,"age")  
def duration() :
    balance= pd.Series
    box=charge.ints(donnees,"duration")     
def pdays() :
    balance= pd.Series
    box=charge.ints(donnees,"pdays")   
def previous() :
    balance= pd.Series
    box=charge.ints(donnees,"previous")  
def campaign() :
    balance= pd.Series
    box=charge.ints(donnees,"campaign")  
def day() :
    balance= pd.Series
    box=charge.ints(donnees,"day")                        

def cible_int() :
    box =charge.cible_int(donnees,"deposit")   
def int_cat() :
     box=charge.int_cat(donnees)
def int_int()  :   
    table=charge.int_int(donnees)
def charger() :
     return charge.chargement(donnees)  
