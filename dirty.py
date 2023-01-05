from pydub import AudioSegment
import os
import random
i = 501


def con(file1, file2):
    sound1 = AudioSegment.from_wav(file1)
    sound2 = AudioSegment.from_wav(file2)
    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = sound1.overlay(sound2, position=2000)
    # save the result
    output.export(f"{file1[:-4]}_ac" + ".wav", format="wav")

def converter(file1,i):
    m4a_file = file1
    wav_filename = "m"+i+".wav"
    track = AudioSegment.from_file(m4a_file, format='m4a')
    file_handle = track.export(wav_filename, format='wav')


if __name__ == '__main__':
    i=1
    directory1 =r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\males"
    directory2 =r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\females"
    dir3=r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\sound"
    voice_list=[]
    for file_voice in os.listdir(dir3):
        fv= os.path.join(dir3,file_voice)
        voice_list.append(fv)
    # file_to_mix = "airplane-fly-over-02a.wav"
    for filename in os.listdir(directory1):
        if filename[-4:] == ".wav":
            print(filename)
            f = os.path.join(directory1, filename)

            # if "_" not in f:
            x=random.randint(0,1)
            if x==1:
                print("yes",i)
                i+=1
                y=random.randint(0, len(voice_list)-1)
                con(f,voice_list[y])
                print(f)
                os.remove(f)



