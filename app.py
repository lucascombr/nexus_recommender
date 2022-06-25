#loading libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

#Ignore Warnings
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 

# Define Pasta Arquivos
os.chdir(os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Read CSV
df = pd.read_csv('df_disease.csv')

# Read Model
x_file = open('bayes.pkl', "rb")
clf = pickle.load(x_file)

#Cria Dicionario de Categorias
categorias_code = pd.read_csv('categorias_code.csv')

#Le dataframe de precautions
precautions = pd.read_csv('df_precaucao.csv')

#Cria cabeçalho
st.image('Nexus_logo.png', use_column_width=True)
#st.title("Recomendação de Tratamentos")
st.subheader("Informações Pessoais")

#Cria quantidade de sintomas
num = []
for x in (range(1,18)):
    num.append(x)

nome = st.text_input('Qual seu nome?')
if len(nome) == 0:
     st.write(f'<h1 style="color:#ff0000;font-size:13px;">{"Por favor, digite um nome."}</h1>', unsafe_allow_html=True)
idade = st.text_input(f'Olá {nome} qual a sua idade?')
if len(idade) == 0:
     st.write(f'<h1 style="color:#ff0000;font-size:13px;">{"Por favor, digite uma idade."}</h1>', unsafe_allow_html=True)
sex = st.radio(
     "Qual o seu sexo?",
     ('Male', 'Female'))

st.subheader("Sintomas")

colunas_sintomas = ['Symptom_1', 'Symptom_2', 'Symptom_3']
sintomas = []
for x in colunas_sintomas:
    for y in df[x].unique():
        sintomas.append(y)
sintomas = np.unique(sintomas)
sintomas_list= []
for x in sintomas:
    for y,z in zip(categorias_code['Categorias_nome'],categorias_code['Traducao']):
        if x == y:
            sintomas_list.append(z)
sintomas = sintomas_list

sintomas_choica = st.multiselect(
    "Seleciona os sintomas:", sintomas)

if len(sintomas_choica) != 3:
     st.write(f'<h1 style="color:#ff0000;font-size:13px;">{"Por favor, selecione três sintomas."}</h1>', unsafe_allow_html=True)

if len(nome)!= 0 and len (idade) !=0 and len(sintomas_choica) ==3:
    if st.button('Recomendar tratamento'):
        if len(sintomas_choica) == 3:
           # for x in range(len(sintomas_choica)+1,18):
           #     sintomas_choica.append("Nenhum")
            sintomas_choica = pd.DataFrame(sintomas_choica)      
            sintomas_choica_list= []
            for x in sintomas_choica[0]:
                for y,z in zip(categorias_code['Traducao'],categorias_code['Categorias_code']):
                    if x == y:
                        sintomas_choica_list.append(z)
            sintomas_choica[0] = sintomas_choica_list
            sintomas_choica = pd.pivot_table(sintomas_choica, columns = df.columns[2:-1], aggfunc='first')
            #sintomas_choica['idade']= (idade)
            if sex == 'Male':
                sintomas_choica['sex']= 1
            else:
                sintomas_choica['sex']= 0
            to_predict = sintomas_choica[colunas_sintomas].to_numpy()
            to_predict = -np.sort(-to_predict)
            agesex = sintomas_choica[['sex']].to_numpy()
            to_predict = np.concatenate((to_predict,agesex), axis=1)
            probas = clf.predict_proba(to_predict)
            top_n_lables_idx = np.argsort(-probas, axis=1)[:, :3].reshape(-1,)
            top_n_probs = np.round(np.round(-np.sort(-probas),3)[:, :3]*100,3).reshape(-1,)
            top_n_labels = [clf.classes_[i] for i in top_n_lables_idx]
            results = list(zip(top_n_labels, top_n_probs))
            results = pd.DataFrame(results, columns = ['Label','Porcentagem de Chance'])
            results_list= []
            for x in results['Label']:
                for y,z in zip(categorias_code['Categorias_code'],categorias_code['Traducao']):
                    if x == y:
                        results_list.append(z)
            results['Label'] = results_list
            st.markdown("***")
            st.subheader('Vamos la, aqui estão algumas possíveis causas do seus sintomas:')
            for x,z,y in zip(results['Label'],results['Porcentagem de Chance'],['1º','2º','3º']):
                st.write(f'<b style="color:#015303;font-size:20px;"> {y} </b>A porcentagem de chance de você estar com <b style="color:#ff0000;font-size:15px;">{x}</b> é de <b style="color:#ff0000;font-size:15px;">{z}%</b>. <br><i>Caso tenha interesse em saber mais sobre {x}, clique em:</i> <a href="https://www.google.com/search?q={x}">Saiba Mais</a>  ', unsafe_allow_html=True)
                lista = list(precautions.loc[precautions.Disease.isin([x]),['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].melt()['value'])
                st.write(f'&emsp; Vamos passar algumas precauções que você pode adotar: <ul> <li> {lista[0]} </li> <li> {lista[1]} </li> <li> {lista[2]} </li> <li> {lista[3]} </li> </ul>', unsafe_allow_html=True)
        st.write('Obrigado por usar o Nexus, desejamos melhoras.')
    else:
        st.write('Clique em Recomendar Tratamento')
else:
    st.write('Verifique os campos em vermelho')

