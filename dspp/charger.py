import pandas as pd 
import numpy as pn 
import matplotlib.pyplot as plt 
import seaborn as sn 
import plotly.express as pl
import streamlit as st
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
class charge :
    source = r"D:\fraude\donnees\bank.csv"
    st.cache_data
    def __init__(self,s:pd.Series) -> pd.Series:
         serie =self.s
    st.cache_data  
    def chargement(source) :
         return pd.read_csv(source)
    st.cache_data
    def univarie(source) :
          data=pd.read_csv(source)
          for i,col in enumerate(data.select_dtypes("object")) :
               donnee= data[col].value_counts().reset_index()
               fig= pl.bar(donnee,x=col,y="count",text_auto=".s",title=col)
               st.plotly_chart(fig)
          for i,col in enumerate(data.select_dtypes("int64")) :
              figs =pl.histogram(data,x=data[col],title=col)
              st.plotly_chart(figs) 
    st.cache_data          
    def cat_cat(source) :
          data=pd.read_csv(source)
          data = data.select_dtypes("object")
          col1=data.columns[:5].tolist()
          col2 =data.columns[-5:].tolist()
          for col in data : 
               if col in col1 :
                    for cols in data : 
                         if cols in col2 :
                             da=data.groupby(col)[cols].value_counts().reset_index()
                             fig=pl.bar(da,x=col,y="count",color=cols,text_auto="",title=f"relation entre {col} et {cols}")
                             st.plotly_chart(fig)
                             
    st.cache_data                       
    def ints(source,series) :
          c= series
          data=pd.read_csv(source)
          figs =pl.box(data,x=data[c],title=series)
          st.plotly_chart(figs)
    st.cache_data     
    def cible_int(source,xi)  :
         data= pd.read_csv(source)
         for col in data.select_dtypes("int64") :
              fig=pl.box(data,x=xi,y=col,title=f"relation entre {xi} et {col}")
              st.plotly_chart(fig)
    st.cache_data                          
    def int_cat(source)  :
         data= pd.read_csv(source)
         for col in data.select_dtypes("object") :
               if col!="deposit" :
                   for cols in data.select_dtypes("int64") :
                      fig=pl.box(data,x=col,y=cols,title=f"relation entre {col} et {cols}")
                      st.plotly_chart(fig)
    st.cache_data       
    def int_int(source)  :
          data=pd.read_csv(source)
          data = data.select_dtypes("int64")
          tab1,tab2=st.tabs(["📈 Chart", "🗃 Data"] )
          tab1.subheader("relation")
          tab1.pyplot(sn.pairplot(data))
          tab2.subheader("relation")
          tab2.table(data.corr(method="spearman").style.background_gradient())
    st.cache_data     
    def balance( balance) :
          if balance <=10 :
               return 1
          elif ((balance >10) & (balance<=25)) :
               return 2
          else :
               return 3  
    st.cache_data         
    def train_test(source,cible):
         data= pd.read_csv(source)
         for col in data.select_dtypes("object") :
               data[col] = data[col].astype("category").cat.codes
         x=data.copy()
         y= x.pop(cible)
         x_train,x_test ,y_train,y_test =train_test_split(x,y,test_size=0.2,stratify=y,random_state=0)
         return x_train,x_test,y_train,y_test
    st.cache_data
    def feature(source,cible) :
          data= pd.read_csv(source)
          data=data.drop(cible,axis=1)
          return data
    st.cache_data
    def pdays(pdays) :
          if pdays <=200 :
               return 1
          elif ((pdays >200) & (pdays<=400)) :
               return 2
          else :
               return 3




                 
               
        

     

   
              
       