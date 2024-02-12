import pickle
import time
import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the data
df = pd.read_csv('survey_lung _cancer.csv')

# Mengonversi data ke dalam DataFrame pandas
df = pd.DataFrame.from_records(df)

# Mengganti nama kolom
df = df.rename(columns={
  'GENDER': 'Gender',
  'AGE': 'Age',
  'SMOKING': 'Smoking',
  'YELLOW_FINGERS': 'YellowFingers',
  'ANXIETY': 'Anxiety',
  'PEER_PRESSURE': 'PeerPressure',
  'CHRONIC DISEASE': 'ChronicDisease',
  'WHEEZING': 'Wheezing',
  'FATIGUE': 'Fatigue',
  'ALLERGY': 'Allergy',
  'ALCOHOL CONSUMING': 'AlcoholConsuming',
  'COUGHING': 'Coughing',
  'SHORTNESS OF BREATH': 'ShortnessOfBreath',
  'SWALLOWING DIFFICULTY': 'SwallowingDifficulty',
  'CHEST PAIN': 'ChestPain',
  'LUNG_CANCER': 'LungCancer',
})

df.drop_duplicates(inplace=True)

# Label encoding
columns_to_encode = ['Gender', 'Smoking', 'YellowFingers', 'Anxiety', 'PeerPressure',
       'ChronicDisease', 'FATIGUE ', 'ALLERGY ', 'Wheezing',
       'AlcoholConsuming', 'Coughing', 'ShortnessOfBreath',
       'SwallowingDifficulty', 'ChestPain', 'LungCancer']

encoder = LabelEncoder()

for column in columns_to_encode:
    df[column] = encoder.fit_transform(df[column])

X = df.drop(['LungCancer'], axis=1)
y = df['LungCancer']

# Oversampling of Minority Class
X, y = RandomOverSampler().fit_resample(X,y)

# Load the model
model = pickle.load(open('svc_model_over_normal_tun.pkl', 'rb'))
model_info = pickle.load(open('model_info.pkl', 'rb'))
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))

# Melakukan prediksi menggunakan model pada data yang telah diproses
y_predict = model.predict(X)
accuracy = accuracy_score(y, y_predict)
accuracy = accuracy*100

df_final = X
df_final['LungCancer'] = y

#STREAMLIT

# mengatur title web browser
st.set_page_config(
    page_title = "Lung Cancer Prediction App",
    page_icon = ":lung:"
)

# judul webpage
st.title("Lung Cancer Prediction App")

# _ : italic, ** : bold
st.write(f"**_Model's Accuracy_** : :green[**{accuracy}**]% (:red[_Do not copy  outright_])")
st.write("")

tab1, tab2 = st.tabs(["Single-predict", "Multi-predict"])

with tab1 :
    st.sidebar.header("**User Input** Sidebar")

    age = st.sidebar.number_input(label=":violet[**Age**]", min_value=df_final['age'].min(), max_value=df_final['age'].max())
    st.sidebar.write(f":orange[Min] value: :orange[**{df_final['age'].min()}**], :red[Max] value: :red[**{df_final['age'].max()}**]")
    st.sidebar.write("")

    gender_sb = st.sidebar.selectbox(label=":blue[**Gender**]", options=["Male", "Female"])
    st.sidebar.write("")
    st.sidebar.write("")
    if gender_sb == "Male":
      Gender = 1
    elif gender_sb == "Female":
      Gender = 0
    # -- Value 0: Female
    # -- Value 1: Male
    
    smoking_sb = st.sidebar.selectbox(label=":blue[**Smoking**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if smoking_sb == "Yes":
      Smoking = 1
    elif smoking_sb == "No":
      Smoking = 0
    # -- Value 0: No
    # -- Value 1: Yes
    
    yellow_sb = st.sidebar.selectbox(label=":blue[**Yellow Fingers**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if yellow_sb == "Yes":
      YellowFingers = 1
    elif yellow_sb == "No":
      YellowFingers = 0
    # -- Value 0: No
    # -- Value 1: Yes

    anxiety_sb = st.sidebar.selectbox(label=":blue[**Anxiety**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if anxiety_sb == "Yes":
      Anxiety = 1
    elif anxiety_sb == "No":
      Anxiety = 0
    # -- Value 0: No
    # -- Value 1: Yes

    peer_sb = st.sidebar.selectbox(label=":blue[**Peer Pressure**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if peer_sb == "Yes":
      PeerPressure = 1
    elif peer_sb == "No":
      PeerPressure = 0
    # -- Value 0: No
    # -- Value 1: Yes

    chronic_sb = st.sidebar.selectbox(label=":blue[**Chronic Disease**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if chronic_sb == "Yes":
      ChronicDisease = 1
    elif chronic_sb == "No":
      ChronicDisease = 0
    # -- Value 0: No
    # -- Value 1: Yes

    fatigue_sb = st.sidebar.selectbox(label=":blue[**Fatigue**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if fatigue_sb == "Yes":
      FATIGUE = 1
    elif fatigue_sb == "No":
      FATIGUE = 0
    # -- Value 0: No
    # -- Value 1: Yes
    
    allergy_sb = st.sidebar.selectbox(label=":blue[**Allergy**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if allergy_sb == "Yes":
      ALLERGY = 1
    elif allergy_sb == "No":
      ALLERGY = 0
    # -- Value 0: No
    # -- Value 1: Yes

    wheezing_sb = st.sidebar.selectbox(label=":blue[**Wheezing**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if wheezing_sb == "Yes":
      Wheezing = 1
    elif wheezing_sb == "No":
      Wheezing = 0
    # -- Value 0: No
    # -- Value 1: Yes

    alcohol_sb = st.sidebar.selectbox(label=":blue[**Alcohol Consuming**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if alcohol_sb == "Yes":
      AlcoholConsuming = 1
    elif alcohol_sb == "No":
      AlcoholConsuming = 0
    # -- Value 0: No
    # -- Value 1: Yes

    cough_sb = st.sidebar.selectbox(label=":blue[**Coughing**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if cough_sb == "Yes":
      Coughing = 1
    elif cough_sb == "No":
      Coughing = 0 
    # -- Value 0: No
    # -- Value 1: Yes

    short_sb = st.sidebar.selectbox(label=":blue[**Shortness of Breath**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if short_sb == "Yes":
      ShortnessOfBreath = 1
    elif short_sb == "No":
      ShortnessOfBreath = 0
    # -- Value 0: No
    # -- Value 1: Yes

    Swallowing_sb = st.sidebar.selectbox(label=":blue[**Swallowing Difficulty**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if Swallowing_sb == "Yes":
      SwallowingDifficulty = 1
    elif Swallowing_sb == "No":
      SwallowingDifficulty = 0
    # -- Value 0: No
    # -- Value 1: Yes

    ChestPain_sb = st.sidebar.selectbox(label=":blue[**Chest Pain**]", options=["Yes", "No"])
    st.sidebar.write("")
    st.sidebar.write("")
    if ChestPain_sb == "Yes":
      ChestPain = 1
    elif ChestPain_sb == "No":
      ChestPain = 0
    # -- Value 0: No
    # -- Value 1: Yes

    data = {
        'Age' : age,
        'Gender': gender_sb,
        'Smoking': smoking_sb,
        'Yellow Fingers': yellow_sb,
        'Anxiety': anxiety_sb,
        'Peer Pressure': peer_sb,
        'Chronic Disease': chronic_sb,
        'Fatigue': fatigue_sb,
        'Allergy': allergy_sb,
        'Wheezing': wheezing_sb,
        'Alcohol Consuming': alcohol_sb,
        'Coughing': cough_sb,
        'Shortness of Breath': short_sb,
        'Swallowing Difficulty': Swallowing_sb,
        'Chest Pain': ChestPain_sb,
    }     

    preview_df = pd.DataFrame(data, index=['input'])

    st.subheader("User Input as Dataframe")
    st.write("")
    st.dataframe(preview_df)
    st.write("")

    result = ":violet[-]"

    predict_btn = st.button("**Predict**", type="primary")

    st.write("")
    if predict_btn:
        inputs = [[Age, Gender, Smoking, YellowFingers, Anxiety, PeerPressure, ChronicDisease, FATIGUE, ALLERGY, Wheezing, AlcoholConsuming, Coughing, ShortnessOfBreath, SwallowingDifficulty, ChestPain]]
        inputs = scaler_model.transform(inputs)
        prediction = model.predict(inputs)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101) :
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()
            
        if prediction == 0:
            result = ":green[**Kanker Non-Paru**]"
        elif prediction == 1:
            result = ":red[**Kanker Paru**]"
        
        st.write("")
        st.write("")
        st.subheader("Prediction :")
        st.subheader(result)

with tab2 :
    st.sidebar.header("**User Input** Sidebar")

    upload_file = st.sidebar.file_uploader(label=":violet[**Upload File**]", type=['csv', 'xlsx'])
    st.sidebar.write("")

    if upload_file is not None:
        try:
            df = pd.read_csv(upload_file)
        except Exception as e:
            print(e)
            df = pd.read_excel(upload_file)

        st.subheader("User Input as Dataframe")
        st.write("")
        st.dataframe(df)
        st.write("")

        df = df.dropna()
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)

        inputs = df.drop(columns=['id'])
        inputs = scaler_model.transform(inputs)

        prediction = model.predict(inputs)

        df['Prediction'] = prediction

        st.subheader("Prediction :")
        st.write("")
        st.dataframe(df)
        st.write("")
