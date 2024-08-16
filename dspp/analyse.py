import matplotlib.pyplot as plt
import numpy as pn
import seaborn as sn
import plotly.express as pl
from charger import charge
import streamlit as st
import pandas as pd
def univ() :
    data= charge.univarie(r"D:\fraude\donnees\bank.csv")    
def cat_cat() :
    data= charge.cat_cat(r"D:\fraude\donnees\bank.csv")     
def balance() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","balance") 
def age() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","age")  
def duration() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","duration")     
def pdays() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","pdays")   
def previous() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","previous")  
def campaign() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","campaign")  
def day() :
    balance= pd.Series
    box=charge.ints(r"D:\fraude\donnees\bank.csv","day")                        

def cible_int() :
    box =charge.cible_int(r"D:\fraude\donnees\bank.csv","deposit")   
def int_cat() :
     box=charge.int_cat(r"D:\fraude\donnees\bank.csv")
def int_int()  :   
    table=charge.int_int(r"D:\fraude\donnees\bank.csv")
def charger() :
     return charge.chargement(r"D:\fraude\donnees\bank.csv")  
