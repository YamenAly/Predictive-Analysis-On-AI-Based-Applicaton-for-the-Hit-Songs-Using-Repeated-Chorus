# Predictive-Analysis-On-AI-Based-Applicaton-for-the-Hit-Songs-Using-Repeated-Chorus
### 1. Introduction
 A music hook(or the catchy part of the song) is simply the element of a song that draws the listener's attention. It can be a riff in the song or merely a distinguishing sound, but it is usually the opening few lines of a song's repeating chorus. Creating a hook is a typical songwriting method. The best hooks will stay in your head for days. Songwriters and producers generally assume that writing a successful hook is what makes a song popular. We investigated this hypothesis in this study by generating a data set of choruses from popular musicians and using supervised Machine Learning (ML) techniques to predict the popularity of their works only based on audio attributes extracted from the chorus.

### 2. Dataset
 As far as we know there is not an publicly available dataset of the song hooks, therefore we had to collect our own.
 1. Using the billboard.py API we collected the names of all the popular and unpopular songs from the top artists on the Billboard website
 2. We downloaded the full songs using the Youtube-DLP python library
 3. Extracted the chorus using the pychorus library
 4. Extracted the audio features from the hooks using the Librosa library
 
 ### 3. Audio Feature Engineering
 Using youtube-dlp and the names of popular artists and their popular and unpopular songs, we retrieve the complete audio files from Youtube. We select the first published studio album version for songs with many versions, such as remakes or remixes. The pychorus is then used to extract 15 seconds of repeated chorus from each song track. 'Pychorus' primary idea is to extract comparable structures from the frequency spectrum. It is essentially a type of unsupervised learning, and the performance of this extraction is open to interpretation. However, in this work, we presume they are correct. We examined a few familiar songs empirically, and the extracted chorus matched our perception. Librosa was used to extract many audio elements from the chorus. For the investigation, we chose 11 key spectral features. Table 1 lists their names, descriptions, and dimensions. The dimensions of the feature set for each track are also a function of, where t is a function of the chorus's I ti duration (in this example 15s) and the soundtrack's sampling rate i. Because we don't want the duration or sampling rate to affect the prediction, we converted the raw dimensions into seven statistics: min, mean, median, max, standard deviation, skew, and kurtosis. After the modification, each music track generates 525 audio features in total.
 
 ![image](https://user-images.githubusercontent.com/84917734/207155094-1743055d-7816-48d4-97a2-1f065fa2a6ae.png)

### 4. The Final Dataset 
In this dataset, we collected total of 80 unique artists with total of 828 popular and unpopular songs from billboard hit songs on the HOT-100 quarter-end chart between 2006 and 2022. Therefore, the final dataset in this report has a 795 rows × 525 columns audio feature matrix and a 795 × 1 response vector.

### 5. Web App
Finally, a web app was created using streamlit to insert songs and predict whether the song is going to be popular or not.
