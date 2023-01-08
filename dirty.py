from pydub import AudioSegment
import os
import random
i = 501

# file 1 is our original sound
def con(file1, file2):
    # used to create an AudioSegment object from a WAV file
    sound1 = AudioSegment.from_wav(file1)
    sound2 = AudioSegment.from_wav(file2)
    # mix sound2 with sound1, starting at 5000ms into sound1)
    output = sound1.overlay(sound2, position=2000)
    # save the result
    output.export(f"{file1[:-4]}_ac" + ".wav", format="wav")

if __name__ == '__main__':
    i=1
    directory1 =r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\males"
    directory2 =r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\females"
    #The files with the noise that we added in the background
    dir3=r"C:\Users\User\PycharmProjects\gender_detection\voice_gender_detection-master\data\sound"
    voice_list=[]
    for file_voice in os.listdir(dir3):
        #os.path.join is a function in the os.path module
        # that can be used to join two or more paths together to form a single path.
        fv= os.path.join(dir3,file_voice)
        voice_list.append(fv)

    for filename in os.listdir(directory1):
        # Skip json files
        if filename[-4:] == ".wav":
            print(filename)
            f = os.path.join(directory1, filename)

            x=random.randint(0,1)
            if x==1:
                print("yes",i)
                i+=1
                y=random.randint(0, len(voice_list)-1)
                #Takes a random file that we will added
                con(f,voice_list[y])
                print(f)
                #Deleting the "clean" file
                os.remove(f)





