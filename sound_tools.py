from pydub import AudioSegment
from pydub.playback import play
from os import listdir
import random
import json
import pandas as pd
import numpy as np
import librosa
import torchaudio
import os, nltk, random, json
from speechbrain.pretrained import EncoderClassifier

def exportfile(newAudio, time1, time2, filename, i):
    # Exports to a wav file in the current path.
    newAudio2 = newAudio[time1:time2]
    g = os.listdir()
    if filename[0:-4] + '_' + str(i) + '.wav' in g:
        filename2 = str(i) + '_segment' + '.wav'
        print('making %s' % (filename2))
        newAudio2.export(filename2, format="wav")
    else:
        filename2 = str(i) + '.wav'
        print('making %s' % (filename2))
        newAudio2.export(filename2, format="wav")

    return filename2


def audio_time_features(filename):
    # recommend >0.50 seconds for timesplit
    timesplit = 0.50
    hop_length = 512
    n_fft = 2048

    y, sr = librosa.load(filename)
    duration = float(librosa.core.get_duration(y))

    # Now splice an audio signal into individual elements of 100 ms and extract
    # all these features per 100 ms
    segnum = round(duration / timesplit)
    deltat = duration / segnum
    timesegment = list()
    time = 0

    for i in range(segnum):
        # milliseconds
        timesegment.append(time)
        time = time + deltat * 1000

    newAudio = AudioSegment.from_wav(filename)
    filelist = list()

    for i in range(len(timesegment) - 1):
        filename = exportfile(newAudio, timesegment[i], timesegment[i + 1], filename, i)
        filelist.append(filename)

        featureslist = np.array([0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0])

        # save 100 ms segments in current folder (delete them after)
        for j in range(len(filelist)):
            try:
                features = featurize(filelist[i])
                featureslist = featureslist + features
                os.remove(filelist[j])
            except:
                print('error splicing')
                featureslist.append('silence')
                os.remove(filelist[j])

        # now scale the featureslist array by the length to get mean in each category
        featureslist = featureslist / segnum

        return featureslist



def featurize(wavfile):
    #initialize features 
    hop_length = 512
    n_fft=2048
    #load file 
    y, sr = librosa.load(wavfile)
    #extract mfcc coefficients 
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc) 
    #extract mean, standard deviation, min, and max value in mfcc frame, do this across all mfccs
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


def add_512_features():
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    MALES_PATH = r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\males"
    FEMALES_PATH = r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\females"
    male_files = listdir(MALES_PATH)
    female_files = listdir(FEMALES_PATH)
    random.shuffle(male_files)
    random.shuffle(female_files)
    min_amount = 1000
    boys = []
    girls = []
    count = 0
    for file in male_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = np.append(featurize(f"{MALES_PATH}/{file}"),audio_time_features(f"{MALES_PATH}/{file}"))
            signal, fs =torchaudio.load(f"{MALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            boys.append(features.tolist() + embedding.tolist()) #720
            # boys.append(features.tolist()) #208
            # boys.append(embedding.tolist()) #512
            count += 1
    
    count = 0
    for file in female_files:
        if file[-3:] == 'wav':
            if count >= min_amount:
                break
            features = np.append(featurize(f"{FEMALES_PATH}/{file}"),audio_time_features(f"{FEMALES_PATH}/{file}"))
            signal, fs =torchaudio.load(f"{FEMALES_PATH}/{file}")
            embeddings = classifier.encode_batch(signal)
            embeddings = embeddings.detach().cpu().numpy()
            embedding = embeddings[0][0]
            girls.append(features.tolist() + embedding.tolist()) #720
            # girls.append(features.tolist()) #208
            # girls.append(embedding.tolist()) #512
            count += 1

    print("boys: ", len(boys))
    print("girls: ", len(girls))

    json_obj = {"males": boys, "females": girls}
    with open('data/males_females_audio.json', 'w') as outfile:
        json.dump(json_obj, outfile)

if __name__ == '__main__':

    add_512_features()

