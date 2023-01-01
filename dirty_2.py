from audiomentations import Compose, AddGaussianNoise, TimeMask, PitchShift, BandStopFilter
import numpy as np
from audiomentations.augmentations.mp3_compression import Mp3Compression
import soundfile as sf  # save data as wav file
import pydub  # convert wav format to mp3
import os
import glob
from pathlib import Path
import audio2numpy as a2n

path = r'C:\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data'
new_path = path + '\\name_for_output_folder'  # for every type of augment change name here
os.makedirs(new_path)  # only once creates directory
mp3_pre_augment = glob.glob('/path/to/folder/with/mp3/*mp3')  # switch with path to location of mp3 files
# m4a_pre_augment = glob.glob(r'C:\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\dirty')  # switch with path to location of mp3 files

# Change p = 0 for augmentations you dont want to use and p = 1 to augmentation you want
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.03, p=0),
    PitchShift(min_semitones=-6, max_semitones=8, p=0),
    BandStopFilter(min_center_freq=60, max_center_freq=2500, min_bandwidth_fraction=0.1, max_bandwidth_fraction=0.4,
                   p=1)
])

print('----------- Augmenting... ----------------')
for m4a in mp3_pre_augment:
    # if m4a[-4:] == ".wav":
    print("!11111"+m4a)
    filename = Path(m4a).stem
    fullpath = r'C:\Users\galco\PycharmProjects\deepLearning\voice_gender_detection-master\data\dirty'+ filename + '.m4a'
    x, sr = a2n.audio_from_file(fullpath)
    augmented_samples = augment(samples=x, sample_rate=48000)
    sf.write(new_path + '/' + filename + 'blabla.wav', augmented_samples, 48000)
print('----------- Augmenting complete. ----------\n\n')


# for wav in wav_files:
#     # print(wav)
#     m4a_file = os.path.splitext(wav)[0] + '.m4a'
#     sound_2 = pydub.AudioSegment.from_wav(wav)
#     sound_2.export(m4a_file, format="m4a")
#     os.remove(wav)
# print('----------- exporting to m4a complete. ----------\n\n')