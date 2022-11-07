import numpy as np
import pickle
import pandas as pd
import streamlit as st
import os
import glob
import librosa
import soundfile as sf
import numpy as np
from yt_dlp import YoutubeDL
from requests import get
from pychorus import find_and_output_chorus
from scipy.stats import skew , kurtosis
import base64
import warnings
warnings.filterwarnings('ignore')
from annotated_text import annotated_text




file_ = open("/home/yamen/Projects/Technocolabs/giphy.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
      """

st.markdown(hide_menu_style, unsafe_allow_html=True)

# Button alignment
col1, col2, col3 = st.sidebar.columns([1,1,1])
if col1.button("About"):
    st.sidebar.text("ML App TO Predict Billboard Hot Songs")
    st.sidebar.text("Developed By Yamen Aly") 
if col2.button("INFO"):
    st.sidebar.text("Model 1: Random Forest Classifier")
    st.sidebar.text("Test Accuracy : 91%") 
    st.sidebar.text("Model 2: Logistic Regression")
    st.sidebar.text("Test Accuracy : 64%")
    st.sidebar.text("Model 3: Support Vector Machine")
    st.sidebar.text("Test Accuracy : 91%")
    st.sidebar.text("Model 4: Decision Tree")
    st.sidebar.text("Test Accuracy : 82%")

# Select Classifiers            
classifier_name = st.sidebar.selectbox("Select Classifier", ("Random Forest", "Logistic Regression", "Support Vector Machine", "Decision Tree"))

def get_model(classifier_name):
    if classifier_name == "Random Forest":
        pickle_in = open("rfmodel.pkl","rb")
        classifier=pickle.load(pickle_in)
    elif classifier_name == "Logistic Regression":
        pickle_in = open("lrmodel.pkl","rb")
        classifier=pickle.load(pickle_in)
    elif classifier_name == "Support Vector Machine":
        pickle_in = open("svmmodel.pkl","rb")
        classifier=pickle.load(pickle_in)
    elif classifier_name == "Decision Tree":
        pickle_in = open("dtmodel.pkl","rb")
        classifier=pickle.load(pickle_in)
    return classifier

classifier = get_model(classifier_name)        

def predict_note_authentication(filename):
    

    def statistics(list, feature, columns_name, data):
      i = 0
      for ele in list:
        _skew = skew(ele)
        columns_name.append(f'{feature}_kew_{i}')
        min = np.min(ele)
        columns_name.append(f'{feature}_min_{i}')
        max = np.max(ele)
        columns_name.append(f'{feature}_max_{i}')
        std = np.std(ele)
        columns_name.append(f'{feature}_std_{i}')
        mean = np.mean(ele)
        columns_name.append(f'{feature}_mean_{i}')
        median = np.median(ele)
        columns_name.append(f'{feature}_median_{i}')
        _kurtosis = kurtosis(ele)
        columns_name.append(f'{feature}_kurtosis_{i}')
        
        i += 1
        data.append(_skew)
        data.append(min)
        data.append(max)
        data.append(std)
        data.append(mean)
        data.append(median)
        data.append(_kurtosis)
      return data

    from librosa.feature.spectral import mfcc
    def extract_features(audio_path, title):

      
      columns_name = ['Title']
      data.append(title)
      x , sr = librosa.load(filename)

      chroma_stft = librosa.feature.chroma_stft(x, sr)
      stft = statistics(chroma_stft, 'chroma_stft', columns_name, data)
      #stft1.append(stft)

      chroma_cqt = librosa.feature.chroma_cqt(x, sr)
      cqt = statistics(chroma_cqt, 'chroma_cqt', columns_name, data)
      #cqt1.append(cqt)

      chroma_cens = librosa.feature.chroma_cens(x, sr)
      cens = statistics(chroma_cens, 'chroma_cens', columns_name, data)
      #cens1.append(cens)

      mfcc = librosa.feature.mfcc(x, sr)
      mf = statistics(mfcc, 'mfcc', columns_name, data)
      #mf1.append(mf)

      rms = librosa.feature.rms(x, sr)
      rm = statistics(rms, 'rms', columns_name, data)
      #rm1.append(rm)

      spectral_centroid = librosa.feature.spectral_centroid(x, sr)
      centroid = statistics(spectral_centroid, 'spectral_centroid', columns_name, data)
      #centroid1.append(centroid)

      spectral_bandwidth = librosa.feature.spectral_bandwidth(x, sr)
      bandwidth = statistics(spectral_bandwidth, 'spectral_bandwidth', columns_name, data)
      #bandwidth1.append(bandwidth)

      spectral_contrast = librosa.feature.spectral_contrast(x, sr)
      contrast = statistics(spectral_contrast, 'spectral_contrast', columns_name, data)
      #contrast1.append(contrast)

      spectral_rolloff = librosa.feature.spectral_rolloff(x, sr)
      rolloff = statistics(spectral_rolloff, 'spectral_rolloff', columns_name, data)
      #rolloff1.append(rolloff)

      tonnetz = librosa.feature.tonnetz(x, sr)
      tonnetz = statistics(tonnetz, 'tonnetz', columns_name, data)
      #tonnetz1.append(tonnetz)

      zero_crossing_rate = librosa.feature.zero_crossing_rate(x, sr)
      zero = statistics(zero_crossing_rate, 'zero_crossing_rate', columns_name, data)
      #zero1.append(zero)

      return data , columns_name

    data = []
    columns_name = []
    #for i , chorus in enumerate(chorus_audio):
    data , columns_name = extract_features(f"{filename}", filename) 

    nnn = []
    for i in range(0, len(data), len(columns_name)):
        nnn.append(data[i:i + 519])
    # creating dataframe
    df2 = pd.DataFrame(nnn, columns=columns_name)
    #df2 

    modelfeature = ['chroma_stft_std_2',
     'chroma_stft_kurtosis_3',
     'chroma_stft_std_6',
     'chroma_stft_kew_8',
     'chroma_stft_kurtosis_8',
     'chroma_stft_kew_11',
     'chroma_cqt_kew_3',
     'chroma_cqt_min_4',
     'chroma_cqt_kurtosis_9',
     'chroma_cens_kew_5',
     'chroma_cens_kurtosis_10',
     'mfcc_min_0',
     'mfcc_kurtosis_2',
     'mfcc_max_5',
     'mfcc_kew_12',
     'mfcc_mean_13',
     'mfcc_min_14',
     'spectral_contrast_mean_0',
     'spectral_contrast_min_1',
     'spectral_contrast_kurtosis_1',
     'spectral_contrast_max_4',
     'spectral_rolloff_kew_0',
     'tonnetz_mean_2',
     'tonnetz_kurtosis_4',
     'zero_crossing_rate_kurtosis_0']

    df = df2[modelfeature]
    st.text("Model Features")
    st.dataframe(df, 1000, 100)
    d  = df.stack().to_numpy()
    df = pd.Series(np.all(df))
    prediction=classifier.predict([d])
    print(prediction)
    return prediction

def with_features(chroma_stft_std_2,chroma_stft_kurtosis_3,chroma_stft_std_6,chroma_stft_kew_8,chroma_stft_kurtosis_8,
                  chroma_stft_kew_11,chroma_cqt_kew_3,chroma_cqt_min_4,chroma_cqt_kurtosis_9,chroma_cens_kew_5,
                  chroma_cens_kurtosis_10,mfcc_min_0,mfcc_kurtosis_2,mfcc_max_5,mfcc_kew_12,mfcc_mean_13,
                  mfcc_min_14,spectral_contrast_mean_0,spectral_contrast_min_1,spectral_contrast_kurtosis_1,
                  spectral_contrast_max_4,spectral_rolloff_kew_0,tonnetz_mean_2,tonnetz_kurtosis_4,
                  zero_crossing_rate_kurtosis_0):
    
    pred=classifier.predict([[chroma_stft_std_2,chroma_stft_kurtosis_3,chroma_stft_std_6,chroma_stft_kew_8,chroma_stft_kurtosis_8,
                  chroma_stft_kew_11,chroma_cqt_kew_3,chroma_cqt_min_4,chroma_cqt_kurtosis_9,chroma_cens_kew_5,
                  chroma_cens_kurtosis_10,mfcc_min_0,mfcc_kurtosis_2,mfcc_max_5,mfcc_kew_12,mfcc_mean_13,
                  mfcc_min_14,spectral_contrast_mean_0,spectral_contrast_min_1,spectral_contrast_kurtosis_1,
                  spectral_contrast_max_4,spectral_rolloff_kew_0,tonnetz_mean_2,tonnetz_kurtosis_4,
                  zero_crossing_rate_kurtosis_0]])
    print(pred)
    return pred



def main():
    
    st.markdown(
     f'<img src="data:image/gif;base64,{data_url}" alt="music gif">',
    unsafe_allow_html=True,
    )
    st.title("""
        Hit Songs Prediction With Four Classifiers
        """)
    
    html_temp = """
    <body>
    <div style="background-color:#ff4b4b;padding:10px">
    <h2 style="color:white;text-align:center;">ML App Predicts Hit Songs</h2>
    </div>
    </body>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    filename = st.file_uploader("Upload Song",accept_multiple_files = False, type=["wav"])

    url = st.sidebar.text_input("Enter URL : ", key='url')

    
    if filename is not None:
      result=""
      if st.button("Predict"):
        result=predict_note_authentication(filename)
        if result == 1:
            st.success('IT IS A POPULAR SONG.')
        else:
            st.success("IT IS AN UNPOPULAR SONG.")    
        st.success('The predicted output label is {}'.format(result))
    
    if filename is None:

        if(not os.path.exists(os.path.join(os.getcwd(), "mp3dir"))):
            os.mkdir(os.path.join(os.getcwd(), "mp3dir"))
        
        if(not os.path.exists(os.path.join(os.getcwd(), "wavdir"))):
            os.mkdir(os.path.join(os.getcwd(), "wavdir"))
        
        SAVE_PATH = '/'.join(os.getcwd().split('/')[:3]) + '/mp3dir'
        file_path = SAVE_PATH + '/audio.%(ext)s'                  #'/%(title)s.%(ext)s'
        YDL_OPTIONS = {'outtmpl':file_path,'format':'bestaudio/best', 'ignoreerrors': True, 'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192', 
        }], 'noplaylist' : 'True'}

        with YoutubeDL(YDL_OPTIONS) as ydl:
            audio = ydl.download((url,))
        file = os.path.join(os.getcwd()+"/mp3dir", os.listdir(os.getcwd() + "/mp3dir")[0])
        fname = os.listdir(os.getcwd() + "/mp3dir")[0]
        chor_path= SAVE_PATH + "/../wavdir/chorus.wav"                              #f"/../wavdir/{fname}.wav"    
        find_and_output_chorus(file,chor_path, 15)
        path = SAVE_PATH + "/../wavdir"
        for filename in glob.glob(os.path.join(path, '*.wav')):
          audio_file1 = open(filename, 'rb')       
          audio_byte = audio_file1.read()
          st.audio(audio_byte, format="audio/wav")
          result=""
          result=predict_note_authentication(filename)
          if result == 1:
            st.success('IT IS A POPULAR SONG.')
          else:
            st.success("IT IS AN UNPOPULAR SONG.")    
          st.success('The predicted output label is {}'.format(result))

        #Experimental Purpose
        st.text("-------------Experimental Purpose----------")
        st.text("Still In Under-Development Phase ......................")
        st.sidebar.text("------ Experimental Purpose -------")

        chroma_stft_std_2 = st.sidebar.number_input(label="chroma_stft_std_2",step=1.,format="%.2f")

        chroma_stft_kurtosis_3 = st.sidebar.number_input(label="chroma_stft_kurtosis_3",step=1.,format="%.2f")

        chroma_stft_std_6 = st.sidebar.number_input(label="chroma_stft_std_6",step=1.,format="%.2f")

        chroma_stft_kew_8 = st.sidebar.number_input(label="chroma_stft_kew_8",step=1.,format="%.2f")

        chroma_stft_kurtosis_8 = st.sidebar.number_input(label="chroma_stft_kurtosis_8",step=1.,format="%.2f")

        chroma_stft_kew_11 = st.sidebar.number_input(label="chroma_stft_kew_11",step=1.,format="%.2f")

        chroma_cqt_kew_3 = st.sidebar.number_input(label="chroma_cqt_kew_3",step=1.,format="%.2f")

        chroma_cqt_min_4 = st.sidebar.number_input(label="chroma_cqt_min_4",step=1.,format="%.2f")

        chroma_cqt_kurtosis_9 = st.sidebar.number_input(label="chroma_cqt_kurtosis_9",step=1.,format="%.2f")

        chroma_cens_kew_5 = st.sidebar.number_input(label="chroma_cens_kew_5",step=1.,format="%.2f")

        chroma_cens_kurtosis_10 = st.sidebar.number_input(label="chroma_cens_kurtosis_10",step=1.,format="%.2f")

        mfcc_min_0 = st.sidebar.number_input(label="mfcc_min_0",step=1.,format="%.2f")

        mfcc_kurtosis_2 = st.sidebar.number_input(label="mfcc_kurtosis_2",step=1.,format="%.2f")

        mfcc_max_5 = st.sidebar.number_input(label="mfcc_max_5",step=1.,format="%.2f")

        mfcc_kew_12 = st.sidebar.number_input(label="mfcc_kew_12",step=1.,format="%.2f")

        mfcc_mean_13 = st.sidebar.number_input(label="mfcc_mean_13",step=1.,format="%.2f")

        mfcc_min_14 = st.sidebar.number_input(label="mfcc_min_14",step=1.,format="%.2f")

        spectral_contrast_mean_0 = st.sidebar.number_input(label="spectral_contrast_mean_0",step=1.,format="%.2f")

        spectral_contrast_min_1 = st.sidebar.number_input(label="spectral_contrast_min_1",step=1.,format="%.2f")

        spectral_contrast_kurtosis_1 = st.sidebar.number_input(label="spectral_contrast_kurtosis_1",step=1.,format="%.2f")

        spectral_contrast_max_4 = st.sidebar.number_input(label="spectral_contrast_max_4",step=1.,format="%.2f")

        spectral_rolloff_kew_0 = st.sidebar.number_input(label="spectral_rolloff_kew_0",step=1.,format="%.2f")

        tonnetz_mean_2 = st.sidebar.number_input(label="tonnetz_mean_2",step=1.,format="%.2f")

        tonnetz_kurtosis_4 = st.sidebar.number_input(label="tonnetz_kurtosis_4",step=1.,format="%.2f")

        zero_crossing_rate_kurtosis_0 = st.sidebar.number_input(label="zero_crossing_rate_kurtosis_0",step=1.,format="%.2f")

        exp_result=""
        if st.button("Experiment"):
            exp_result=with_features(chroma_stft_std_2,chroma_stft_kurtosis_3,chroma_stft_std_6,chroma_stft_kew_8,chroma_stft_kurtosis_8,
            chroma_stft_kew_11,chroma_cqt_kew_3,chroma_cqt_min_4,chroma_cqt_kurtosis_9,chroma_cens_kew_5,
            chroma_cens_kurtosis_10,mfcc_min_0,mfcc_kurtosis_2,mfcc_max_5,mfcc_kew_12,mfcc_mean_13,
            mfcc_min_14,spectral_contrast_mean_0,spectral_contrast_min_1,spectral_contrast_kurtosis_1,
            spectral_contrast_max_4,spectral_rolloff_kew_0,tonnetz_mean_2,tonnetz_kurtosis_4,
            zero_crossing_rate_kurtosis_0)
            
            if exp_result == 1:
              st.success('IT IS A POPULAR SONG.')
            else:
              st.success("IT IS AN UNPOPULAR SONG.")      
    
    # if audio_file is None:
    #     audio_file = audio_file1
    #     with open(os.path.join("SaveDir", audio_file.name), "wb") as f:
    #         f.write(audio_file.getbuffer())  
    
    
    

if __name__=='__main__':
    main()
