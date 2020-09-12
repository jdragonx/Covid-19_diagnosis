import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import keras.backend.tensorflow_backend as tb

from keras.models import load_model
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix

tb._SYMBOLIC_SCOPE.value = True

model_RF = pickle.load(open("model_RF.pkl","rb"))
model_NN = load_model(open("model_NN.h5","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
y_nn = pickle.load(open("y_nn.pkl","rb"))
y_rf = pickle.load(open("y_rf.pkl","rb"))

DATA_URL = (
    "datos_limpios.csv"
)

st.title("Covid-19 patient data, analysis")
st.sidebar.title("Covid-19 data")
st.markdown("""
<style>
body {
    background-color: #65E05A
}
</style>
""",unsafe_allow_html=True)

@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

data = load_data()

option = st.sidebar.radio("Available tasks",options=("Make diagnosis","View average distribution","View model effectiveness"))

if option == "Make diagnosis":
    st.sidebar.markdown("Data insertion")
    st.markdown("⚕️ Diagnosis for Covid-19 using Machine Learning and simple blood tests")
    diag = []
    diag.append(st.sidebar.number_input("hematocrit", min_value=-10., max_value=10., value=-1.57168221473694))
    diag.append(st.sidebar.number_input("hemoglobin", min_value=-10., max_value=10., value=-0.774212002754211))
    diag.append(st.sidebar.number_input("platelets", min_value=-10., max_value=10., value=1.4296674728393601))
    diag.append(st.sidebar.number_input("mean_platelet_volume", min_value=-10., max_value=10., value=-1.6722217798233))
    diag.append(st.sidebar.number_input("red_blood_cells", min_value=-10., max_value=10., value=-0.850035011768341))
    diag.append(st.sidebar.number_input("lymphocytes", min_value=-10., max_value=10., value=-0.00573804322630167))
    diag.append(st.sidebar.number_input("mean_corpuscular_hemoglobin_concentration_mchc", min_value=-10., max_value=10., value=3.3310706615448))
    diag.append(st.sidebar.number_input("leukocytes", min_value=-10., max_value=10., value=0.364550471305847))
    diag.append(st.sidebar.number_input("basophils", min_value=-10., max_value=10., value=-0.223766505718231))
    diag.append(st.sidebar.number_input("mean_corpuscular_hemoglobin_mch", min_value=-10., max_value=10., value=0.178174987435341))
    diag.append(st.sidebar.number_input("eosinophils", min_value=-10., max_value=10., value=1.0186250209808299))
    diag.append(st.sidebar.number_input("mean_corpuscular_volume_mcv", min_value=-10., max_value=10., value=-1.33602428436279))
    diag.append(st.sidebar.number_input("red_blood_cell_distribution_width_rdw", min_value=-10., max_value=10., value=-0.978899121284485))

    result = ''
    
    if st.button('Diagnose with random forest', key=1):
        result = model_RF.predict(np.array(diag).reshape(1,-1))
        result = (str(result[0]))
        result = 'covid '+result
        
    if st.button('Diagnose with neural network', key=2):
        x_nn = scaler.transform(np.array(diag).reshape(1,-1))
        result = model_NN.predict_classes(x_nn)
        if result==0:
            result = 'covid negative'
        else:
            result = 'covid positive'
        
    if result!='':
        if result=='covid negative':
            st.write("Diagnosis completed, result: "+result+'😃')
        if result=='covid positive':
            st.write("Diagnosis completed, result: "+result+'😔')
        
names = ['hematocrit','hemoglobin','platelets','mean_platelet_volume','red_blood_cells','lymphocytes','mean_corpuscular_hemoglobin_concentration_mchc','leukocytes','basophils','mean_corpuscular_hemoglobin_mch','eosinophils','mean_corpuscular_volume_mcv','red_blood_cell_distribution_width_rdw']
figs = []
hema_avg = data.to_numpy()
hema_avg_n = hema_avg[hema_avg[:,2]=="negative"]
hema_avg_n = np.mean(hema_avg_n[:,6:],axis=0)
hema_avg_p = hema_avg[hema_avg[:,2]=="positive"]
hema_avg_p = np.mean(hema_avg_p[:,6:],axis=0)
hema = pd.DataFrame([hema_avg_p,hema_avg_n],columns=names)
hema = hema.transpose()
hema.columns = ['covid positive','covid negative']
hemadisp = hema
hemadisp = hemadisp
hema = hema.transpose()
hema['diagnosis'] = hema.index
for meas in names:
    figs.append(px.bar(hema, x='diagnosis',y=meas,color='diagnosis', height=500))

if option == "View average distribution":
    st.subheader("Average value of each meassument")
    mode = st.sidebar.radio("Visualization",("Table","Bar"),key=1)
    if (mode=="Table"):
        st.dataframe(hemadisp,height=500,width=600)
    if (mode=="Bar"):
        for fig in figs:
            st.plotly_chart(fig)

total_acc = y_rf == data['sars_cov_2_exam_result'].to_numpy()
rf_acc = accuracy_score(data['sars_cov_2_exam_result'].to_numpy(),y_rf)
rf_acc = int(rf_acc*100)
pier_rf = pd.DataFrame(y_rf)[0].value_counts()
pier_rf = pd.DataFrame(pier_rf)
pier_rf.columns = ['values']

piep_rf = data['sars_cov_2_exam_result'].value_counts()
piep_rf = pd.DataFrame(piep_rf)
piep_rf.columns = ['values']


total_acc_nn = y_nn == data['sars_cov_2_exam_result'].to_numpy()
nn_acc = accuracy_score(data['sars_cov_2_exam_result'].to_numpy(),y_nn)
nn_acc = int(nn_acc*100)
pier_nn = pd.DataFrame(y_nn)[0].value_counts()
pier_nn = pd.DataFrame(pier_nn)
pier_nn.columns = ['values']

piep_nn = data['sars_cov_2_exam_result'].value_counts()
piep_nn = pd.DataFrame(piep_nn)
piep_nn.columns = ['values']
if option == "View model effectiveness":
    m_view = st.sidebar.radio("Comparision",("Random Forest","Neural Network","Both models"))
    if m_view == "Random Forest":
        st.markdown("<p style='text-align: center'>While training: 100%</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>While testing: 88%</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>With the full data: "+str(rf_acc)+"%</p>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(12,5))
        ax = []
        ax = fig.subplots(1,2)
        ax[0].axis('equal')
        ax[0].pie(piep_rf.values[:,0], labels = piep_rf.index,autopct='%1.2f%%')
        ax[0].set_title('Real distribution',loc='center')
        ax[1].axis('equal')
        ax[1].set_title('Predicted distribution',loc='center')
        ax[1].pie(pier_rf.values[:,0], labels = pier_rf.index,autopct='%1.2f%%')
        st.pyplot()
        
    if m_view == "Neural Network":
        st.markdown("<p style='text-align: center'>While training: 92%</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>While testing: 92%</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>With the full data: "+str(nn_acc)+"%</p>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(12,5))
        ax = []
        ax = fig.subplots(1,2)
        ax[0].axis('equal')
        ax[0].pie(piep_nn.values[:,0], labels = piep_nn.index,autopct='%1.2f%%')
        ax[0].set_title('Real distribution',loc='center')
        ax[1].axis('equal')
        ax[1].set_title('Predicted distribution',loc='center')
        ax[1].pie(pier_nn.values[:,0], labels = pier_nn.index,autopct='%1.2f%%')
        st.pyplot()
        
    if m_view == "Both models":
        st.markdown("<h2 style='text-align: center'>Confusion Matrix</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center'>Random Forest</p>", unsafe_allow_html=True)
        matrix_rf = confusion_matrix(data['sars_cov_2_exam_result'].to_numpy(),y_rf)
        st.dataframe(pd.DataFrame(matrix_rf,columns=["clasified as positive","clasified as negative"],index=["positive","negative"]),width=700)
        st.markdown("<p style='text-align: center'>Neural Network</p>", unsafe_allow_html=True)
        matrix_nn = confusion_matrix(data['sars_cov_2_exam_result'].to_numpy(),y_nn)
        st.dataframe(pd.DataFrame(matrix_nn,columns=["clasified as positive","clasified as negative"],index=["positive","negative"]),width=700)
        st.markdown("<h2 style='text-align: center'>Pie Chart</h2>", unsafe_allow_html=True)
        
        fig = plt.figure(figsize=(12,5))
        ax = []
        ax = fig.subplots(1,2)
        ax[0].axis('equal')
        ax[0].pie(pier_rf.values[:,0], labels = pier_rf.index,autopct='%1.2f%%')
        ax[0].set_title('Pie Chart Random Forest',loc='center')
        ax[1].axis('equal')
        ax[1].set_title('Pie Chart Neural Network',loc='center')
        ax[1].pie(pier_nn.values[:,0], labels = pier_nn.index,autopct='%1.2f%%')
        st.pyplot()