import pandas as pd
import matplotlib.pyplot as pd
import numpy as pn
import seaborn as sn
import plotly.express as pl
from charger import charge
import streamlit as st
import analyse
import donnees
from streamlit_option_menu import option_menu
import machinelearning
from xgboost import XGBClassifier



def main() :
    st.sidebar.header(":red[Me]:blue[nu]",divider='rainbow')
    menu =[":blue[Data]",":blue[analyse]",":blue[MachineLearnig]"]
    option=st.sidebar.radio(":blue[Page]",menu)  
    if option==":blue[Data]" :
          data=  donnees.data
          st.header(":blue[Dat]:red[a]",divider='rainbow')
          st.dataframe(data.sample(n=5,replace=True))
          st.write(f"le dataset contient {data.shape[0]} lignes et {data.shape[1]} colonnes",unsafe_allow_html=True)
          if st.button("statistique") :
                st.table(data.describe().style.background_gradient())
               
          
                           
                 
    elif option ==":blue[analyse]" :
         with st.sidebar :
           options=option_menu("Analyse",["Analyse exploratoire","Tableau de bord"])
         if options =="Analyse exploratoire" :
               analyse.univ()
               if st.button("cat_cat") :
                 analyse.cat_cat()
               if st.button("boxplot")   :
                 col1,col2,col3,col4,col5,col6=st.columns((6),gap="medium")

                 with col1 :
                
                   analyse.balance()
              
                 with col2 :
                     
                    analyse.age() 
                 with col3 :
               
                    analyse.duration() 
                 with col4 :
                
                    analyse.campaign()   
                 with col5 :
                
                    analyse.pdays() 
                                   
                 with col6 :
                      analyse.day()  
               if st.button("cible_int") :
                     analyse.cible_int() 
               if st.button("cat_int") :
                    analyse.int_cat() 
               if st.button("int_int") :
                      analyse.int_int()   
         else :  
             st.sidebar.header(":blue[filtration] :red[par] :") 
             data=analyse.charger()
             deposit= st.sidebar.multiselect("Deposit",data["deposit"].unique(),default=data["deposit"].unique())
             default= st.sidebar.multiselect("Default",data["default"].unique(),default=data["default"].unique())
             job= st.sidebar.multiselect("Job",data["job"].unique(),default=data["job"].unique()) 
             education= st.sidebar.multiselect("Education",data["education"].unique(),default=data["education"].unique()) 
             housing= st.sidebar.multiselect("housing",data["housing"].unique(),default=data["housing"].unique()) 
             marital= st.sidebar.multiselect("marital",data["marital"].unique(),default=data["marital"].unique()) 
             contact= st.sidebar.multiselect("contact",data["contact"].unique(),default=data["contact"].unique()) 
             
             dat=data[(data.deposit.isin(deposit))&(data.default.isin(default))&((data.job.isin(job)))&(data.education.isin(education))&(data.housing.isin(housing))&(data.marital.isin(marital))&(data.contact.isin(contact))]
             st.dataframe(dat)
             col1,col2= st.columns((2),gap="medium")
             with col1 :
                 with st.container(height=480,border=True) :
                   st.header("analyse univariée")
                   with st.expander("categorielle") :
                     cat=st.selectbox("select",dat.select_dtypes("object").columns,on_change=None)
                     tab=dat[cat].value_counts().reset_index() 
                     if dat[cat].nunique()<5 :
                           fig=pl.pie(tab,names=cat,values="count")
                           st.plotly_chart(fig)
                     else :
                         fig=pl.bar(tab,x=cat,y="count",text_auto="")
                         st.plotly_chart(fig)
                   with st.expander("numerique") : 
                        num=st.selectbox("select",dat.select_dtypes("int64").columns,on_change=None)
                        fig=pl.histogram(dat,x=num)
                        st.plotly_chart(fig)
             with col2 : 
                        with st.container(height=480,border=True) :
                             st.header("analyse bivariée")
                             with st.expander("categorielle_numerique") :
                               cats=st.selectbox("select",dat.select_dtypes("object").columns,on_change=None,key=10)
                               nums=st.selectbox("select",dat.select_dtypes("int64").columns,on_change=None,key=11)
                               fig=pl.box(dat,x=cats,y=nums)
                               st.plotly_chart(fig)
             with st.container(height=480,border=True) : 
                 st.header("table de statistique")  
                 st.table(dat.describe().style.background_gradient())               
    elif option==":blue[MachineLearnig]"  :
        opt =st.sidebar.radio(":blue[Sous]:red[-menu]",[":red[model]",":red[prediction]"])
        model = XGBClassifier(max_depth=4,subsample=0.65, n_estimators=1000,learning_rate=0.03, min_child_weight=1)    
        if opt==":red[model]" :
           machinelearning.evaluation(model)
        elif opt==":red[prediction]" :
            st.subheader(":red[predic]:blue[tion]",divider='rainbow')
            data=machinelearning.feature()
            col1,col2=st.columns((2))
            with col1 :
                cont1 =st.container(height=450)
                with cont1 :
                    col3,col4=st.columns((2))
                    with col3 :
                       default=st.selectbox("default",data.default.unique())
                       default=machinelearning.default(default)
                       marital=st.selectbox("marital",data.marital.unique())
                       marital=machinelearning.marital(marital)
                       education=st.selectbox("education",data.education.unique())
                       education=machinelearning.education(education)
                       loan=st.selectbox("loan",data.loan.unique())
                       loan=machinelearning.loan(loan)
                    with col4 :
                       contact=st.selectbox("contact",data.contact.unique())
                       contact=machinelearning.contact(contact)
                       housing=st.selectbox("housing",data.housing.unique())
                       housing=machinelearning.housing(housing)
                       month=st.selectbox("month",data.month.unique())
                       month=machinelearning.month(month)
                       poutcome=st.selectbox("poutcome",data.poutcome.unique())
                       poutcome=machinelearning.poutcome(poutcome)
            with col2 :
                cont2 =st.container(height=450)
                with cont2 :
                    col6,col7=st.columns((2))
                    with col6 :
                          job=st.selectbox("job",data.job.unique())
                          job=machinelearning.job(job)
                          balance=st.number_input("balance",min_value=-6487,max_value=81204)
                          balance=machinelearning.balance(balance)
                          age=st.number_input("age",min_value=18,max_value=95)
                          day=st.number_input("day",min_value=1,max_value=31)
                    with col7 :
                         duration=st.number_input("duration",min_value=1,max_value=3881)
                         pdays=st.number_input("pdays",min_value=-1,max_value=854)
                         pdays=machinelearning.pdays(pdays)
                         campaign=st.number_input("campaign",min_value=1,max_value=63)
                         previous= st.number_input("previous",min_value=0,max_value=58) 
                         
            if st.button("prediction") :
                  
                 st.write(machinelearning.prediction(age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome),unsafe_allow_html=True)           
                                         
if __name__== "__main__" :
     main()
     

