import numpy as np
from keras.utils import Sequence
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

class FERGenerator(Sequence):
    def __init__(self, fer_reader, x_set, y_set, batch_size):
        self.fer_reader = fer_reader
        self.x = x_set
        self.y_set = y_set
        self.batch_size = batch_size
        self.y = fer_reader.generate_emotions(y_set)
        self.datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, rotation_range=10,
                             width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
        self.datagen.fit(self.x)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x_modified = [self.datagen.random_transform(item) for item in batch_x]
        return np.array(batch_x_modified), batch_y
    
    def on_epoch_end(self):
        self.y = self.fer_reader.generate_emotions(self.y_set)
