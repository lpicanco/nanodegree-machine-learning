import pandas as pd
import numpy as np
from keras.utils import to_categorical

class FERReader:
    def __init__(self, fer_path, fer_plus_path):
        self.fer_path = fer_path
        self.fer_plus_path = fer_plus_path
        self.emotion_columns = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]
        self.emotion_count = len(self.emotion_columns)

    def read(self):
        df_fer2013 = pd.read_csv(self.fer_path)
        df_ferplus = pd.read_csv(self.fer_plus_path)

        df = self.__join(df_fer2013, df_ferplus)
        df = self.__clean(df)
        self.train_set, self.test_set, self.validation_set = self.__split(df)
        
        X_train = self.__convert_pixels(self.train_set)
        X_validation = self.__convert_pixels(self.validation_set)
        X_test = self.__convert_pixels(self.test_set)
        return (X_train, X_validation, X_test)

    def generate_emotions(self, df):
        return self.__encode_emotions(df)

    def __join(self, df_fer2013, df_ferplus):
        df_ferplus_columns = self.emotion_columns + ["Usage"]
        df_joined_data = pd.concat([df_fer2013.pixels, df_ferplus[df_ferplus_columns]], axis=1)
        return pd.DataFrame(columns=["pixels"] + df_ferplus_columns, data=df_joined_data)

    def __clean(self, df):
        num_votes = df[self.emotion_columns].sum(axis=1)

        #Removing rows where the total votes are <1.
        df.drop(df[num_votes < 1].index, inplace=True)

        # transform votes to probability(between 0 and 1)
        df[self.emotion_columns] = df[self.emotion_columns].div(num_votes, axis=0)

        return df

    def __split(self, df):
        train_set = df[df.Usage == 'Training']
        test_set = df[df.Usage == 'PrivateTest']
        validation_set = df[df.Usage == 'PublicTest']        
        return (train_set, test_set, validation_set)

    def __convert_pixels(self, df):
        imgs = [np.asarray(row.split(" "), dtype=np.float32) / 255 for _, row in enumerate(df.pixels)]
        return np.array(imgs).reshape(len(imgs),48,48,1)

    def __encode_emotions(self, df):
        emotions = [to_categorical(np.random.choice(self.emotion_count, p=row), self.emotion_count) for _, row in df[self.emotion_columns].iterrows()]
        return np.array(emotions).reshape(len(emotions), self.emotion_count)





