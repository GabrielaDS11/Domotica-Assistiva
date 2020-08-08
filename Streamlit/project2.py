import streamlit as st
import pandas as pd
import numpy as np 
#import ploty.express as px
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

def loadData():
    df = pd.read_csv("smart_houses.csv")
    return df

def preprocessing(df):

    #X = df.iloc[:, [0,1,2,3,4,5,6,7,9,10,11,12,14]].values
    #y = df.iloc[:, -3].values
    le = LabelEncoder()
    #df['familia'] = le.fit_transform(df['familia'].values)
    df['classe'] = le.fit_transform(df['classe'].values)
    X = df.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,14,18]].values
    y = df.iloc[:, -3].values #tipo_sh
    y = le.fit_transform(y.flatten())
    #The flatten() function is used to get a copy of an given array collapsed into one dimension

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    return X_train, X_test, y_train, y_test

#Training Decesion Tree for Classificação
@st.cache(suppress_st_warning=True)
def decisionTree(X_train, X_test, y_train, y_test):
    tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    return score, report, tree

#Training KNN Classifier
@st.cache(suppress_st_warning=True)
def Knn_Classifier(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)

    return score, report, clf

#Training Gaussian Model
@st.cache(suppress_st_warning=True)
def gaussian(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred)
    
    return score, report, gnb  

@st.cache(suppress_st_warning=True)
def svclinear(X_train, X_test, y_train, y_test):
    svc_model = LinearSVC(max_iter = 2500, random_state=0)
    svc_model.fit(X_train, y_train)
    y_pred = svc_model.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) *100
    report = classification_report(y_test, y_pred)

    return score, report, svc_model

def accept_user_data():
    n_moradores = st.number_input("Qual o número de moradores",1)
    idosos = st.number_input("Qual o número de idosos",1)
    def_loc = st.number_input("Deficientes Locomoção",1)
    def_b_v = st.number_input("Baixa Visão",1)
    def_cog = st.number_input("Cognitivo",1)
    def_aud = st.number_input("Auditivo",1)
    comodos = st.number_input("numero de comodos",1)
    r_anual = st.number_input("Renda Anual",1)
    renda = (r_anual/n_moradores)
    if renda <= 15800:
        classe = 'baixa'
    elif renda > 15800 and renda <=32500:
        classe = 'media'
    else:
        renda = 'alta'
    A = st.number_input("Automoção",1)
    B = st.number_input("Assistência",1)
    C = st.number_input("Saúde",1)
    D = st.number_input("Comunicação",1)
    E = st.number_input("Lazer",1)
    instalacao = st.number_input("Instalação",1)

    user_prediction_data = np.array([n_moradores, idosos,def_loc,def_b_v,
                                    def_cog,def_aud,comodos,r_anual,renda,A,B,C,
                                    D,E,instalacao])

    return user_prediction_data

def main():
    st.title("Smart Houses Predições")
    data = loadData()
    X_train, X_test, y_train, y_test = preprocessing(data)

    if st.checkbox("Show Raw Data"):
        st.subheader("Showing raw data...")
        st.write(data.head())

    #ML Section
    choose_model = st.sidebar.selectbox("Choose the ML Model",
                                        ["NONE", "Decision Tree","K-Nearest Neighbours","Gaussian",
                                        "SVC Linear Model"])
    
    if(choose_model == "Decision Tree"):
        score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Decision Tree model is: ")
        st.write(score,"%")
        st.text("Report of Decision tree model is: ")
        st.write(report)

        try:
            if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data()     
                if st.button("SUBMIT"):
                    user_prediction_data = user_prediction_data.reshape(1,-1)
                    pred = tree.predict(user_prediction_data)
                    st.write("The Predicted Class is: ", (pred)) 
        except:
            pass
   
        
    elif(choose_model == "K-Nearest Neighbours"):
        score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
        st.text("Accuracy of K-Nearest Neighbour model is: ")
        st.write(score,"%")
        st.text("Report of K-Nearest Neighbour model is: ")
        st.write(report)

        try:
            if(st.checkbox("Want to predict on your own Input? It is recommended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data() 		
                pred = clf.predict(user_prediction_data)
                st.write("The Predicted Class is: ", le.inverse_transform(pred)) # Inverse transform to get the original dependent value. 
        except:
            pass

    elif(choose_model == "Gaussian"):
        score, report, gnb = gaussian(X_train, X_test, y_train, y_test)
        st.text("Accuracy of Gaussian Model is: ")
        st.write(score,"%")
        st.text("Report of Gaussian Model is: ")
        st.write(report)

        try:
            if (st.checkbox("Want to predict on your own Input? It is recomended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data()
                pred = gnb.predict(user_prediction_data)
                st.write("The Predicted Class is:", le.inverse_transform(pred))
        except:
            pass

    elif(choose_model == "SVC Linear Model"):
        score, report, svc_model = svclinear(X_train, X_test, y_train, y_test)
        st.text("Accuracy of SVC Linear Model is: ")
        st.write(score, "%")
        st.text("Report of SVC Linear Model is: ")
        st.write(report)
        
        try:
            if (st.checkbox("Want to predict on your own Input? It is recomended to have a look at dataset to enter values in below tabs than just typing in random values")):
                user_prediction_data = accept_user_data()
                pred = gnb.predict(user_prediction_data)
                st.write("The Predicted Class is:", le.inverse_transform(pred))
        except:
            pass


if __name__ == "__main__":
	main()

    