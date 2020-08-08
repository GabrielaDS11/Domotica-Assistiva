import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def main():
    """ Machine Learning Dataset Explorer"""
    st.title("Machine Learning Dataset Explorer")
    st.subheader("Simple Data Science Explorer with Streamlit")
    
    html_temp = """ 
    <div style="background-color:tomato;">
    <p style="color:white; font-size: 50px">Frase aleatória</p>
    <div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Escolhar um arquivo", filenames)
        return os.path.join(folder_path,selected_filename)

    filename = file_selector()
    st.info("Você escolheu {}". format(filename))

    #Ler os dados
    df = pd.read_csv(filename)
    
    # Mostrar o dataset
    if st.checkbox("Mostrar DataSet"):
        number = st.number_input("Número de linhas para visualizar", 5,10)
        st.dataframe(df.head(number))

    #Mostrar colunas
    if st.button("Nomes das Colunas"):
        st.write(df.columns)

    #Mostrar formatos
    if st.checkbox("Formato do Dataset"):
        st.write(df.shape)
        data_dim = st.radio("Show Dimension By",("Rows","Columns"))
        if data_dim == 'Columns':
            st.text("Número de Colunas")
            st.write(df.shape[1])
        elif data_dim == "Rows":
            st.text("Número de linhas")
            st.write(df.shape[0])
        else:
            st.write(df.shape)

    #Escolher colunas
    if st.checkbox("Selecione as colunas desejadas"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Escolha", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    #Mostrar valores
    if st.button("Valores"):
        st.text("Valores em classes")
        st.write(df.iloc[:,0].value_counts()) #moradores
        st.write(df.iloc[:,1].value_counts()) #idosos
        st.write(df.iloc[:,-1].value_counts()) #crianças
        st.write(df.iloc[:,-2].value_counts()) #familias

    #Mostrar Datatypes
    if st.button("DataTypes"):
        st.write(df.dtypes)

    #Mostrar sumário
    if st.checkbox("Sumário"):
        st.write(df.describe().T)

    #Visualização
    st.subheader("Visualização dos dados")
    #Corelação
    #Seaborn
    if st.checkbox("Seaborn Plot"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot
    #Count plot
    if st.checkbox("Plot of Value Counts"):
        st.text("Value Counts By Target")
        all_columns_names = df.columns.tolist()
        primary_col = st.selectbox("Primary Columm to GroupBy",all_columns_names)
        selected_columns_names = st.multiselect("Select Columns",all_columns_names)
        if st.button("Plot"):
            st.text("Generate Plot")
            if selected_columns_names:
                vc_plot = df.groupby(primary_col)[selected_columns_names].count()
            else:
                vc_plot = df.iloc[:,-1].value_counts()
            st.write(vc_plot.plot(kind="bar"))
            st.pyplot()
    #Pie chart
    if st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        selected_column= st.selectbox("Selecione a coluna desejada", all_columns_names)
        if st.button("Gerar Pie Plot"):
            st.success("Gerando um Pie Plot")
            st.write(df[selected_column].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    #Plot customizado
    st.subheader("Plot Customizado")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Selecione o tipo de plot",['area','bar','line','hist','box','kde'])
    selected_columns_names = st.multiselect("Selecione as colunas", all_columns_names)

    if st.button("Gerar Plot"):
        st.success("Gerando plot de {} para {}".format(type_of_plot,selected_columns_names))

        if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.area_chart(cust_data)

        elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.bar_chart(cust_data)

        elif type_of_plot == 'line':
            cust_data = df[selected_columns_names]
            st.line_chart(cust_data)
        
        elif type_of_plot:
            cust_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(cust_plot)
            st.pyplot()

if __name__ == '__main__':
    main()

