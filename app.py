import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pickle import load

@st.cache
def classify_fault(model, df):
    temp = model.predict(df)
    pred_y = np.argmax(temp, axis=1)
    encoder = LabelEncoder()
    pred_class_y = encoder.inverse_transform(pred_y)
    return pred_class_y


# this is the main function in which we define our webpage
def main():

    #from PIL import Image
    #image = Image.open('taipower.png')

    # display the front end aspect
    st.title('Transmission Line Fault Type Classification App')
    st.header('By TPRI HV Lab Dr. Cheng-Chung Li, Shuo-Fu Hong, Wei-Chih Liang')

    col1, col2 = st.columns(2)
    col1.write('The models are built by the simulation data generated from RTDS,\
                and their goal is to classify \
                the expected fault type of a transmission line.')

    col2.image('https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Taiwan_Power_Company_Seal.svg/220px-Taiwan_Power_Company_Seal.svg.png',
             caption='Taipower', width = 466, use_column_width= 'auto')

    image = Image.open('system.png')
    st.image(image, caption='select one system')

    st.subheader('Select System: ')
    system_list = ['a', 'b']
    select_system = st.radio("Pick one system to classify", system_list)

    # following lines create boxes in which user can enter data required to make prediction

    st.subheader('Input Features')
    IAang = st.slider('IAang', float(data['Min'].loc[0]), float(data['Max'].loc[0]), -9.762292, step = 0.01)
    IAang2 = st.slider('IAang2', float(data['Min'].loc[1]), float(data['Max'].loc[1]), 170.201531, step = 0.01)
    IAmag = st.slider('IAmag', float(data['Min'].loc[2]), float(data['Max'].loc[2]), 0.219938, step = 0.01)
    IAmag2 = st.slider('IAmag2', float(data['Min'].loc[3]), float(data['Max'].loc[3]), 0.219956, step = 0.01)
    IBang = st.slider('IBang', float(data['Min'].loc[4]), float(data['Max'].loc[4]), -129.734629, step = 0.01)
    IBang2 = st.slider('IBang2', float(data['Min'].loc[5]), float(data['Max'].loc[5]), 50.229159, step = 0.01)
    IBmag = st.slider('IBmag', float(data['Min'].loc[6]), float(data['Max'].loc[6]), 0.219811, step = 0.01)
    IBmag2 = st.slider('IBmag2', float(data['Min'].loc[7]), float(data['Max'].loc[7]), 0.219829, step = 0.01)
    ICang = st.slider('ICang', float(data['Min'].loc[8]), float(data['Max'].loc[8]), 110.280106, step = 0.01)
    ICang2 = st.slider('ICang2', float(data['Min'].loc[9]), float(data['Max'].loc[9]), -69.756059, step = 0.01)
    ICmag = st.slider('ICmag', float(data['Min'].loc[10]), float(data['Max'].loc[10]), 0.219967, step = 0.01)
    ICmag2 = st.slider('ICmag2', float(data['Min'].loc[11]), float(data['Max'].loc[11]), 0.219985, step = 0.01)
    VAang1 = st.slider('VAang1', float(data['Min'].loc[12]), float(data['Max'].loc[12]), -1.187182, step = 0.01)
    VAang2 = st.slider('VAang2', float(data['Min'].loc[13]), float(data['Max'].loc[13]), -1.189900, step = 0.01)
    VAmag1 = st.slider('VAmag1', float(data['Min'].loc[14]), float(data['Max'].loc[14]), 181.429819, step = 0.01)
    VAmag2 = st.slider('VAmag2', float(data['Min'].loc[15]), float(data['Max'].loc[15]), 181.383600, step = 0.01)
    VBang1 = st.slider('VBang1', float(data['Min'].loc[16]), float(data['Max'].loc[16]), -121.155129, step = 0.01)
    VBang2 = st.slider('VBang2', float(data['Min'].loc[17]), float(data['Max'].loc[17]), -121.157849, step = 0.01)
    VBmag1 = st.slider('VBmag1', float(data['Min'].loc[18]), float(data['Max'].loc[18]), 181.383244, step = 0.01)
    VBmag2 = st.slider('VBmag2', float(data['Min'].loc[19]), float(data['Max'].loc[19]), 181.337034, step = 0.01)
    VCang1 = st.slider('VCang1', float(data['Min'].loc[20]), float(data['Max'].loc[20]), 118.841576, step = 0.01)
    VCang2 = st.slider('VCang2', float(data['Min'].loc[21]), float(data['Max'].loc[21]), 118.838858, step = 0.01)
    VCmag1 = st.slider('VCmag1', float(data['Min'].loc[22]), float(data['Max'].loc[22]), 181.494419, step = 0.01)
    VCmag2 = st.slider('VCmag2',float(data['Min'].loc[23]), float(data['Max'].loc[23]), 181.448174, step = 0.01)

    features = {'IAang': IAang, 'IAang2': IAang2,'IAmag': IAmag, 'IAmag2': IAmag2,
                'IBang': IBang, 'IBang2': IBang2, 'IBmag': IBmag, 'IBmag2': IBmag2,
                'ICang': ICang, 'ICang2': ICang2, 'ICmag': ICmag, 'ICmag2': ICmag2,
                'VAang1': VAang1, 'VAang2': VAang2, 'VAmag1': VAmag1, 'VAmag2': VAmag2,
                'VBang1': VBang1, 'VBang2': VBang2, 'VBmag1': VBmag1, 'VBmag2': VBmag2,
                'VCang1': VCang1, 'VCang2': VCang2, 'VCmag1': VCmag1, 'VCmag2': VCmag2
                }
    features_df = pd.DataFrame([features])

    features_A = {'IAang': IAang, 'IAang2': IAang2,'IAmag': IAmag, 'IAmag2': IAmag2,
                'IBang': IBang, 'IBang2': IBang2, 'IBmag': IBmag, 'IBmag2': IBmag2,
                'ICang': ICang, 'ICang2': ICang2, 'ICmag': ICmag, 'ICmag2': ICmag2
                }
    features_V = {'VAang1': VAang1, 'VAang2': VAang2, 'VAmag1': VAmag1, 'VAmag2': VAmag2,
                'VBang1': VBang1, 'VBang2': VBang2, 'VBmag1': VBmag1, 'VBmag2': VBmag2,
                'VCang1': VCang1, 'VCang2': VCang2, 'VCmag1': VCmag1, 'VCmag2': VCmag2
                }

    features_df_A = pd.DataFrame([features_A])
    features_df_V = pd.DataFrame([features_V])
    st.caption('Current Features')
    st.dataframe(features_df_A)
    st.caption('Voltage Features')
    st.dataframe(features_df_V)

    # load the model
    @st.cache
    model = load(open('model.pkl', 'rb'))
    # load the scaler
    st.cache
    scaler = load(open('scaler.pkl', 'rb'))

    feature_df = scaler.transform(features_df)

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        fault_type = ''
        prediction = classify_fault(model, features_df)

        if str(prediction) == 'AG':
            fault_type = 'AG'
        elif str(prediction) == 'BG':
            fault_type = 'BG'
        elif str(prediction) == 'CG':
            fault_type = 'CG'
        elif str(prediction) == 'AB':
            fault_type = 'AB'
        elif str(prediction) == 'AC':
            fault_type = 'AC'
        elif str(prediction) == 'BC':
            fault_type = 'BC'
        elif str(prediction) == 'ABG':
            fault_type = 'ABG'
        elif str(prediction) == 'ACG':
            fault_type = 'ACG'
        elif str(prediction) == 'BCG':
            fault_type = 'BCG'
        elif str(prediction) == 'ABC':
            fault_type = 'ABC'

        st.subheader('The Prediction: ')
        st.write('The selected system is '+str(select_system) + '.')
        st.write('Based on the input feature values, the predict fault type is '+ str(fault_type)+'.'
        st.balloons()

if __name__ == '__main__':
    main()