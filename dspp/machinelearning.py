import pandas as pd
import numpy as pn
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,roc_auc_score
from sklearn.model_selection import train_test_split,learning_curve,LearningCurveDisplay
from xgboost import XGBClassifier,plot_importance,plot_tree
from charger import charge 
import streamlit as st
import joblib
import shap
st.cache_data()
def train_test() :
    x_train,x_test,y_train,y_test =charge.train_test(r"D:\fraude\donnees\bank.csv","deposit")
    x_train.balance = x_train.balance.map(charge.balance)
    x_test.balance =  x_test.balance.map(charge.balance)
    x_train.pdays = x_train.pdays.map(charge.pdays)
    x_test.pdays =  x_test.pdays.map(charge.pdays)
    return x_train,x_test,y_train,y_test
x_train,x_test,y_train,y_test=train_test()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.cache_data()
def evaluation(ba) :
    model= ba.fit(x_train,y_train)
    st.text(f"accuracy sur les données d'entrainement {model.score(x_train,y_train)}")
    ypred= model.predict(x_test)
    st.text(classification_report(y_test,ypred))
    fig,ax =plt.subplots()
    cm=confusion_matrix(y_test,ypred,labels=model.classes_)
    ax=ConfusionMatrixDisplay(cm,display_labels=model.classes_).plot()
    st.pyplot()
    N,train,val=learning_curve(model,x_train,y_train,train_sizes=pn.linspace(0.1,1.0,10),cv=5,scoring='f1')
    plt.figure(figsize=(10,10))
    plt.plot(N,train.mean(axis=1),label="train")
    plt.plot(N,val.mean(axis=1),label="validation")
    plt.legend()
    st.pyplot()
    if st.button("importance des colonnes") :
        figs,ax=plt.subplots()
        ax= plot_importance(model).plot()
        st.pyplot()
        figx,axs=plt.subplots()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        axs=shap.summary_plot(shap_values, x_test)
        st.pyplot()
st.cache_data()        
def feature()  :
    data=charge.feature(r"D:\fraude\donnees\bank.csv","deposit") 
    return data 
st.cache_data() 
def prediction(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome) :
     model=joblib.load(filename=r"D:\fraude\xgboostx.joblib")
     x=pn.array([age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]).reshape(1,16)
     if model.predict(x)==0 :
              predi=f"Le client va faire son dépôt à temps avec une probabilité de { model.predict_proba(x)[0][0]} "
              return predi
     else :
          predi=f"Le client va faire son dépôt en retard avec une probabilité de { model.predict_proba(x)[0][1]} " 
          return predi        
def default(default) :
    if default=="no" :
        return 0
    else :
        return 1
def loan(laon) :
    if loan=="no" :
        return 0
    else :
        return 1   
def marital(marital) :
     if marital=="divorced" :
        return 0
     elif marital=="married" :
        return 1 
     else :
         return 2
def contact(contact) :
    if contact=="cellular" :
        return 0
    elif contact=="telephone"  :
        return 1  
    else :
        return 0
def housing(housing) :
    if housing=="no" :
        return 0
    else :
        return 1 
def education(education) :
    if education=="primary" :
        return 0
    elif education=="secondary" :
        return 1  
    elif education=="tertiary" :
        return 2 
    else :
        return 3   
def job(job) :
    if job=="admin." :
        return 0
    elif job=="blue-collar" :
        return 1  
    elif job=="entrepreneur" :
        return 2 
    elif job=="housemaid" :
        return 3 
    elif job=="management" :
        return 4 
    elif job=="retired" :
        return 5 
    elif job=="self-employed" :
        return 6 
    elif job=="services"  :
        return 7  
    elif job=="student"  :
        return 8  
    elif job=="technician"  :
        return 9   
    elif job=="unemployed" :
        return 10 
    else :
        return 11  
def month(month) :
    if month=="apr" :
        return 0
    elif month=="aug" :
        return 1  
    elif month=="dec" :
        return 2 
    elif month=="feb" :
        return 3 
    elif month=="jan" :
        return 4 
    elif month=="jul" :
        return 5 
    elif month=="jun" :
        return 6 
    elif month=="mar"  :
        return 7  
    elif month=="may"  :
        return 8  
    elif month=="nov"  :
        return 9   
    elif job=="oct"  :
        return 10  
    else :
        return 11 
def poutcome(poutcome) :
    if poutcome=="failure" :
        return 0
    elif poutcome=="other" :
        return 1  
    elif poutcome=="success" :
        return 2 
    else :
        return 3        
def balance(balance)  :
    balance=charge.balance(balance)  
    return balance 
def pdays(pdays)  :
    pdays=charge.pdays(pdays)  
    return pdays 