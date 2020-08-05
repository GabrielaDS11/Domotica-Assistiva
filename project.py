
import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib as plt 
import plotly.express as px
import seaborn as sns
import altair as alt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder 

st.sidebar.subheader("Inputs")
def user_input_features():
    n_moradores = st.sidebar.slider('Moradores', 1, 12)
    idosos = st.sidebar.slider('Idosos', 0, 5)  
    def_loc = st.sidebar.slider('Def locomoção', 0, 5)  
    def_b_v = st.sidebar.slider('Def baixa visão', 0, 5)
    def_cog = st.sidebar.slider('Def cog', 0, 5)
    def_aud = st.sidebar.slider('Def auditivo',0, 5)
    comodos = st.sidebar.slider('Comodos da casa', 2, 16)
    r_anual = st.sidebar.slider('Renda Anual', 10000, 50000, 200000)
    crianca = st.sidebar.slider('Crianças', 0, 2, 5)

    renda = (r_anual/n_moradores)
    if renda <= 15800:
        classe = 'baixa'

        if idosos != 0:
            A = st.sidebar.slider('Dispositivos de Automação', 30000, 40000, 65500)
            B = st.sidebar.slider("Dispositivos de Assistência", 10000,15000,25000)
            C = st.sidebar.slider('Dispositivos de Saúde', 16000, 20000, 40000)        
            D = st.sidebar.slider('Dispositivos de Comunicação', 1000, 5000, 6500)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 3000, 5000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 20000)

        elif def_loc != 0:
            A = st.sidebar.slider('Dispositivos de Automação', 30000, 40000, 65500)
            B = st.sidebar.slider("Dispositivos de Assistência", 10000,20000,25000)
            C = 0
            D = st.sidebar.slider('Dispositivos de Comunicação', 1000, 5000, 6500)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 3000, 5000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 20000)

        elif def_b_v != 0:
            A = st.sidebar.slider('Dispositivos de Automação', 30000, 50000, 65500)
            B = st.sidebar.slider("Dispositivos de Assistência", 10000,18000,22000)
            C = 0
            D = st.sidebar.slider('Dispositivos de Comunicação', 1000, 5000, 20000)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 5000, 20000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 30000)
        
        elif def_cog != 0 and def_b_v == 0:
            A = st.sidebar.slider('Dispositivos de Automação', 30000, 50000, 65500)
            B = st.sidebar.slider("Dispositivos de Assistência", 10000,20000,30000)
            C = 0        
            D = st.sidebar.slider('Dispositivos de Comunicação', 1000, 5000, 20000)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 5000, 20000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 30000)

        elif def_aud != 0 and def_b_v == 0: 
            A = st.sidebar.slider('Dispositivos de Automação', 30000, 50000, 65500)
            B = st.sidebar.slider("Dispositivos de Assistência", 10000,15000,21000)
            C = 0     
            D = st.sidebar.slider('Dispositivos de Comunicação', 1000, 5000, 20000)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 5000, 20000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 30000)

        else:
            A = 20000
            B = 30000
            C = 0
            D = 50000
            E = 5000
            instalacao = 11000
                
    elif renda > 15800 and renda <=32500:
        classe = 'media'
        if idosos != 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 90000, 100000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 30000, 70000, 90000)
            C = st.sidebar.slider('Dispositivos de Saúde', 40000, 50000, 60000)        
            D = st.sidebar.slider('Dispositivos de Comunicação', 6500, 7000, 8000)
            E = st.sidebar.slider('Dispositivos de Lazer', 5000, 6000, 6500)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 25000)

        elif def_loc != 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 90000, 100000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 25000,30000,38000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 6500, 7000, 8000)
            E = st.sidebar.slider('Dispositivos de Lazer', 5000, 6000, 6500)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 25000)

        elif def_b_v != 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 90000, 100000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 22000,30000,37000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 6500, 7000, 8000)
            E = st.sidebar.slider('Dispositivos de Lazer', 5000, 6000, 6500)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 25000)
        
        elif def_cog != 0 and def_b_v == 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 90000, 100000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 22000,30000,40000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 6500, 7000, 8000)
            E = st.sidebar.slider('Dispositivos de Lazer', 5000, 6000, 6500)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 25000)

        elif def_aud != 0 and def_b_v == 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 90000, 100000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 22000,30000,37000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 6500, 7000, 8000)
            E = st.sidebar.slider('Dispositivos de Lazer', 5000, 6000, 6500)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 25000)

        else:
            A = 80000
            B = 40000
            C = 0
            D = 80000
            E = 6000
            instalacao = 20000
            
    else:
        classe = 'alta'
        if idosos != 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 65500, 50000, 100000)
            B = st.sidebar.slider("Dispositivos de Assistência", 40000,75000,100000)
            C = st.sidebar.slider('Dispositivos de Saúde', 40000, 50000, 80000)        
            D = st.sidebar.slider('Dispositivos de Comunicação', 8000, 9000, 10000)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 5000, 20000)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 30000)

        if def_loc != 0:        
            A = st.sidebar.slider('Dispositivos de Automação', 65500, 50000, 100000)
            B = st.sidebar.slider("Dispositivos de Assistência", 40000,75000,100000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 8000, 9000, 10000)
            E = st.sidebar.slider('Dispositivos de Lazer', 1000, 5000, 20000)
            instalacao = st.sidebar.slider('Instalação', 15000, 20000, 30000)
        
        if def_b_v != 0:
            A = st.sidebar.slider('Dispositivos de Automação', 100000, 50000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 35000, 50000, 80000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 8000, 9000, 10000)
            E = st.sidebar.slider('Dispositivos de Lazer', 8000, 5000, 15000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 30000)

        if def_cog != 0 and def_b_V == 0:
            A = st.sidebar.slider('Dispositivos de Automação', 100000, 50000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 40000, 45000, 60000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 8000, 9000, 10000)
            E = st.sidebar.slider('Dispositivos de Lazer', 8000, 5000, 15000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 30000)

        if def_aud != 0 and def_b_v == 0:
            A = st.sidebar.slider('Dispositivos de Automação', 100000, 50000, 200000)
            B = st.sidebar.slider("Dispositivos de Assistência", 50000, 60000, 90000)
            C = 0      
            D = st.sidebar.slider('Dispositivos de Comunicação', 8000, 9000, 10000)
            E = st.sidebar.slider('Dispositivos de Lazer', 8000, 5000, 15000)
            instalacao = st.sidebar.slider('Instalação', 6000, 15000, 30000)
        
        else:
            A = 32000
            B = 70000
            C = 0
            D = 100000
            E = 7000
            instalacao = 30000

    data = {'n_moradores': n_moradores,
            'idosos': idosos,
            'def_loc': def_loc,
            'def_b_v': def_b_v,
            'def_cog': def_cog,
            'def_aud': def_aud,
            'comodos': comodos,
            'r_anual': r_anual,
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'E': E,
            'instalacao': instalacao,
            'classe': classe,
            'crianca': crianca}
    features = pd.DataFrame(data, index=[0])
    return features

daf = user_input_features()

st.subheader("Input : Parametros")
st.write(daf)

le = LabelEncoder()
daf['classe'] = le.fit_transform(daf['classe'].values)

st.title("Smart Houses")
df = pd.read_csv("smart_houses.csv")

if st.checkbox('Show raw data'):
    st.dataframe(df)

c = alt.Chart(df).mark_circle().encode(
    alt.X(alt.repeat("column"), type='quantitative'),
    alt.Y(alt.repeat("row"), type='quantitative'),
    color='Origin:N'
).properties(
    width=150,
    height=150
).repeat(
    row = ['n_moradores','comodos','preco'],
    column = ['idosos','r_anual','crianca']
).interactive()

if st.checkbox('Show Charts'):
    st.write(c)

#le = LabelEncoder()
df['familia'] = le.fit_transform(df['familia'].values)
df['classe'] = le.fit_transform(df['classe'].values)
target = df['tipo_sh']
cols_to_use = ['n_moradores','idosos','def_loc','def_b_v','def_cog',
            'def_aud','comodos','r_anual','classe','A','B','C',
            'D','E','instalacao','crianca']

X = df[cols_to_use]
y = target

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.30)
gnb = GaussianNB()
pred = gnb.fit(X_treino, y_treino).predict(X_teste)
#prediction = gnb.fit(X_treino, y_treino).predict(daf)
st.write("Naive-Bayes accuracy: ", accuracy_score(y_teste, pred) )
prediction = gnb.fit(X_treino, y_treino).predict(daf)

st.subheader("Tipo de SH")
st.write(prediction)

#score = metrics.accuracy_score(y_teste, pred) * 100
report = classification_report(y_teste, pred)
st.text(report)

y1 = df['preco']
X_train, X_test,y1_train, y1_test = train_test_split(X, y1, test_size=0.3)

from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)
reg.fit(X_train, y1_train)
#reg.fit(X, y1)
y_pred = reg.predict(X_test)
prediction1 = reg.predict(daf)

st.subheader("Preço")
st.write(prediction1)


