
import librosa, pickle, getpass, time, uuid
import torchaudio
from matplotlib import pyplot as plt
from pydub import AudioSegment
import speech_recognition as sr
import os, nltk, random, json
from nltk import word_tokenize
from nltk.classify import apply_features, SklearnClassifier, maxent
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import svm
from sklearn import metrics
from speechbrain.pretrained import EncoderClassifier

from textblob import TextBlob
import numpy as np

cur_dir=os.getcwd()+'/load_dir'
model_dir=os.getcwd()+'/models'
load_dir=os.getcwd()+'/load_dir'

def featurize2(wavfile):
    #initialize features
    hop_length = 512
    n_fft=2048
    # load file ,  y=audio signal , sr = sample rate of the audio data
    y, sr = librosa.load(wavfile)
    # hop_length: This determines the resolution of the MFCCs that are extracted.
    # n_mfcc: the number of MFCCs to return.
    # The function returns a 2D array of MFCCs,
    # where the rows represent the different frames of the audio signal
    # and the columns represent the different MFCC coefficients.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    # The delta features are calculated by taking the difference between the feature vectors of adjacent frames, weighted by a set of coefficients.
    mfcc_delta = librosa.feature.delta(mfcc)
    #extract mean, standard deviation(np.std), min, and max value in mfcc frame, do this across all mfccs
    mfcc_features=np.array([np.mean(mfcc[0]),np.std(mfcc[0]),np.amin(mfcc[0]),np.amax(mfcc[0]),
                            np.mean(mfcc[1]),np.std(mfcc[1]),np.amin(mfcc[1]),np.amax(mfcc[1]),
                            np.mean(mfcc[2]),np.std(mfcc[2]),np.amin(mfcc[2]),np.amax(mfcc[2]),
                            np.mean(mfcc[3]),np.std(mfcc[3]),np.amin(mfcc[3]),np.amax(mfcc[3]),
                            np.mean(mfcc[4]),np.std(mfcc[4]),np.amin(mfcc[4]),np.amax(mfcc[4]),
                            np.mean(mfcc[5]),np.std(mfcc[5]),np.amin(mfcc[5]),np.amax(mfcc[5]),
                            np.mean(mfcc[6]),np.std(mfcc[6]),np.amin(mfcc[6]),np.amax(mfcc[6]),
                            np.mean(mfcc[7]),np.std(mfcc[7]),np.amin(mfcc[7]),np.amax(mfcc[7]),
                            np.mean(mfcc[8]),np.std(mfcc[8]),np.amin(mfcc[8]),np.amax(mfcc[8]),
                            np.mean(mfcc[9]),np.std(mfcc[9]),np.amin(mfcc[9]),np.amax(mfcc[9]),
                            np.mean(mfcc[10]),np.std(mfcc[10]),np.amin(mfcc[10]),np.amax(mfcc[10]),
                            np.mean(mfcc[11]),np.std(mfcc[11]),np.amin(mfcc[11]),np.amax(mfcc[11]),
                            np.mean(mfcc[12]),np.std(mfcc[12]),np.amin(mfcc[12]),np.amax(mfcc[12]),
                            np.mean(mfcc_delta[0]),np.std(mfcc_delta[0]),np.amin(mfcc_delta[0]),np.amax(mfcc_delta[0]),
                            np.mean(mfcc_delta[1]),np.std(mfcc_delta[1]),np.amin(mfcc_delta[1]),np.amax(mfcc_delta[1]),
                            np.mean(mfcc_delta[2]),np.std(mfcc_delta[2]),np.amin(mfcc_delta[2]),np.amax(mfcc_delta[2]),
                            np.mean(mfcc_delta[3]),np.std(mfcc_delta[3]),np.amin(mfcc_delta[3]),np.amax(mfcc_delta[3]),
                            np.mean(mfcc_delta[4]),np.std(mfcc_delta[4]),np.amin(mfcc_delta[4]),np.amax(mfcc_delta[4]),
                            np.mean(mfcc_delta[5]),np.std(mfcc_delta[5]),np.amin(mfcc_delta[5]),np.amax(mfcc_delta[5]),
                            np.mean(mfcc_delta[6]),np.std(mfcc_delta[6]),np.amin(mfcc_delta[6]),np.amax(mfcc_delta[6]),
                            np.mean(mfcc_delta[7]),np.std(mfcc_delta[7]),np.amin(mfcc_delta[7]),np.amax(mfcc_delta[7]),
                            np.mean(mfcc_delta[8]),np.std(mfcc_delta[8]),np.amin(mfcc_delta[8]),np.amax(mfcc_delta[8]),
                            np.mean(mfcc_delta[9]),np.std(mfcc_delta[9]),np.amin(mfcc_delta[9]),np.amax(mfcc_delta[9]),
                            np.mean(mfcc_delta[10]),np.std(mfcc_delta[10]),np.amin(mfcc_delta[10]),np.amax(mfcc_delta[10]),
                            np.mean(mfcc_delta[11]),np.std(mfcc_delta[11]),np.amin(mfcc_delta[11]),np.amax(mfcc_delta[11]),
                            np.mean(mfcc_delta[12]),np.std(mfcc_delta[12]),np.amin(mfcc_delta[12]),np.amax(mfcc_delta[12])])

    return mfcc_features

# add the _segment
def exportfile(newAudio,time1,time2,filename,i):
    #Exports to a wav file in the current path.
    newAudio2 = newAudio[time1:time2]
    g=os.listdir()
    if filename[0:-4]+'_'+str(i)+'.wav' in g:
        filename2=str(uuid.uuid4())+'_segment'+'.wav'
        newAudio2.export(filename2,format="wav")
    else:
        filename2=str(uuid.uuid4())+'.wav'
        newAudio2.export(filename2, format="wav")

    return filename2

def audio_time_features(filename):
    #recommend >0.50 seconds for timesplit
    timesplit=0.50
    hop_length = 512
    n_fft=2048

    # load file ,  y=audio signal , sr = sample rate of the audio data
    y, sr = librosa.load(filename)
    # The function returns the duration of the audio signal
    duration=float(librosa.core.get_duration(y))

    #Now splice an audio signal into individual elements of 100 ms and extract
    #all these features per 100 ms
    segnum=round(duration/timesplit)
    deltat=duration/segnum
    timesegment=list()
    time=0

    for i in range(segnum):
        #milliseconds
        timesegment.append(time)
        time=time+deltat*1000
    # used to create an AudioSegment object from a WAV file
    newAudio = AudioSegment.from_wav(filename)
    filelist=list()

    for i in range(len(timesegment)-1):
        filename=exportfile(newAudio,timesegment[i],timesegment[i+1],filename,i)
        filelist.append(filename)

        featureslist=np.array([0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0,
                               0,0,0,0])

        #save 100 ms segments in current folder (delete them after)
        for j in range(len(filelist)):
            try:
                features=featurize2(filelist[i])
                featureslist=featureslist+features
                os.remove(filelist[j])
            except:
                print('error splicing')
                os.remove(filelist[j])

        #now scale the featureslist array by the length to get mean in each category
        featureslist=featureslist/segnum

        return featureslist

# The classifier is a machine learning model that takes the embeddings as input and predicts a label
# The source argument specifies the source of the pre-trained encoder
# savedir argument specifies the directory where the models are saved.
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",savedir="pretrained_models/spkrec-xvect-voxceleb")
pca_load = pickle.load(open(r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\pca.pkl", 'rb'))
scaler = pickle.load(open(r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\scaler.sav", 'rb'))

def featurize(wavfile):
    # The torchaudio.load function is used to load an audio signal from a file
    # and return a tensor signal
    # and a scalar fs -  which represents the sample rate of the audio signal.
    signal, fs = torchaudio.load(wavfile)
    # algorithm that takes audio signals and extracts useful features
    embeddings = classifier.encode_batch(signal)
    # convert to numpy
    embeddings = embeddings.detach().cpu().numpy()
    # embedding is a single embedding and is assigned the first embedding in the batch
    # which is the first element of the first element of the embeddings array.
    embedding = embeddings[0][0]
    # crate 208 features
    features1 = np.append(featurize2(wavfile),audio_time_features(wavfile))
    features = np.append(features1,embedding.tolist()).reshape(1,-1)
    # It returns a scaled version of the input data, where all the features lie between 0 and 1.
    features = scaler.transform(features)
    ## transform the input data onto the lower-dimensional space
    features = pca_load.transform(features)
    return features


def convert(file):
    if file[-4:] != '.wav':
        filename=file[0:-4]+'.wav'
        os.system('ffmpeg -i %s %s'%(file,filename))
        os.remove(file)
    elif file[-4:] == '.wav':
        filename=file

    return filename

model_list=list()
os.chdir(model_dir)
listdir=os.listdir()

#the wining model
for i in range(len(listdir)):
    if listdir[i][-12:]=='audio.pickle':
        model_list.append(listdir[i])


count=0
errorcount=0

try:
    os.chdir(load_dir)
except:
    os.mkdir(load_dir)
    os.chdir(load_dir)

listdir=os.listdir()
print(os.getcwd())
for i in range(len(listdir)):
    try:
        if listdir[i][-5:] not in ['Store','.json']:
            if listdir[i][-4:] != '.wav':
                if listdir[i][-5:] != '.json':
                    filename=convert(listdir[i])
            else:
                filename=listdir[i]

            if filename[0:-4]+'.json' not in listdir:
                features = featurize(filename).reshape(1,-1)
                os.chdir(model_dir)
                class_list=list()
                model_acc=list()
                deviations=list()
                modeltypes=list()
                for j in range(len(model_list)):
                    modelname=model_list[j]
                    i1=modelname.find('_')
                    name1=modelname[0:i1]
                    i2=modelname[i1:]
                    i3=i2.find('_')
                    name2=i2[0:i3]

                    loadmodel=open(modelname, 'rb')
                    # the wining model
                    model = pickle.load(loadmodel)
                    print(model)
                    loadmodel.close()
                    #model.predict is a method of a machine learning model
                    # that can be used to make predictions on new data.
                    output = str(model.predict(features)[0])
                    #the ans
                    print(output)

                   # check the percentage of the result
                    scores = model.decision_function(features)
                    # print(scores)
                    # print(np.max(features))
                    # print(np.min(features))
                    if scores < 0:
                        l = (abs(np.min(features))) - abs(scores)
                        l1 = abs(np.min(features)) - l
                        l2 = l1 / (abs(np.min(features)))
                        if l2 < 0.5:
                            print("females " , l2 + 0.5)
                        if l2 > 0.5 and l2 < 1:
                            print("females " , l2)
                        if l2 >= 1:
                            print("females ", 1)

                    if scores > 0:
                        l = (np.max(features) - scores)
                        l1 = np.max(features)-l
                        l2 =l1/(np.max(features))
                        if l2 < 0.5:
                            print("males " , l2 + 0.5)
                        if l2 > 0.5 and l2 < 1:
                            print("males " , l2)
                        if l2 >= 1:
                            print("males ", 1)


                    # Create a figure and a subplot
                    fig, ax = plt.subplots()
                    # the range of x and y
                    ax.set_xlim(np.min(features), np.max(features))
                    plt.ylim(np.min(features),np.max(features))
                    ax.scatter(scores, 0)
                    # Show the plot
                    plt.show()

                    classname = output
                    class_list.append(classname)
                    
                    # take the males_females_sc_audio.json data
                    g=json.load(open(modelname[0:-7]+'.json'))
                    model_acc.append(g['accuracy'])
                    deviations.append(g['deviation'])
                    modeltypes.append(g['modeltype'])

                os.chdir(load_dir)
                # open json file for the audio file
                jsonfilename=filename[0:-4]+'.json'
                jsonfile=open(jsonfilename,'w')
                data={
                    'filename':filename,
                    'filetype':'audio file',
                    'class':class_list,
                    'model':model_list,
                    'model accuracies':model_acc,
                    'model deviations':deviations,
                    'model types':modeltypes,
                    'features':features.tolist(),
                    'count':count,
                    'errorcount':errorcount,
                    }
                # add the data for the json
                json.dump(data,jsonfile)
                jsonfile.close()

            count=count+1
    except:
        errorcount=errorcount+1
        count=count+1
