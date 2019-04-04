import pandas as pd
import numpy as np
from keras.utils import to_categorical

class FERReader:
    def __init__(self, fer_path, fer_plus_path):
        self.fer_path = fer_path
        self.fer_plus_path = fer_plus_path
        self.emotion_columns = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt","unknown","NF"]
        self.emotion_count = len(self.emotion_columns)
        self.num_voters = 10

    def read(self):
        df_fer2013 = pd.read_csv(self.fer_path)
        df_ferplus = pd.read_csv(self.fer_plus_path)

        df = self.__join(df_fer2013, df_ferplus)
        df = self.__clean(df)
        df_train, df_test, df_validation = self.__split(df)
        
        X_train = self.__convert_pixels(df_train)
        y_train = self.__encode_emotions(df_train)
        X_test = self.__convert_pixels(df_test)
        y_test = self.__encode_emotions(df_test)
        X_validation = self.__convert_pixels(df_validation)
        y_validation = self.__encode_emotions(df_validation)
        return (X_train, y_train, X_validation, y_validation, X_test, y_test)


    def __join(self, df_fer2013, df_ferplus):
        df_ferplus_columns = self.emotion_columns + ["Usage"]
        df_joined_data = pd.concat([df_fer2013.pixels, df_ferplus[df_ferplus_columns]], axis=1)
        return pd.DataFrame(columns=["pixels"] + df_ferplus_columns, data=df_joined_data)

    def __clean(self, df):
        # Removing rows where the total votes are != num_voters.
        df.drop(df[df[self.emotion_columns].sum(axis=1) != self.num_voters].index, inplace=True)

        # transform votes to probability
        df[self.emotion_columns] = df[self.emotion_columns] / self.num_voters

        # add target column
        df['target'] = df[self.emotion_columns].apply(lambda row: np.random.choice(self.emotion_count, p=row), axis=1)

        # Removing rows where the emotion is unknown/NF.
        columns_to_remove = [self.emotion_columns.index('unknown'), self.emotion_columns.index('NF')]
        df = df[~df['target'].isin(columns_to_remove)]
        self.emotion_columns.remove("unknown")
        self.emotion_columns.remove("NF")
        self.emotion_count = len(self.emotion_columns)
        return df

    def __split(self, df):
        df_train = df[df.Usage == 'Training']
        df_test = df[df.Usage == 'PrivateTest']
        df_validation = df[df.Usage == 'PublicTest']        
        return (df_train, df_test, df_validation)

    def __convert_pixels(self, df):
        imgs = [np.asarray(row.split(" "), dtype=np.float32) / 255 for _, row in enumerate(df.pixels)]
        return np.array(imgs).reshape(len(imgs),48,48,1)

    def __encode_emotions(self, df):
        emotions = [to_categorical(row, self.emotion_count) for _, row in enumerate(df.target)]
        return np.array(emotions).reshape(len(emotions), self.emotion_count)





