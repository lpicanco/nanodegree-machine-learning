from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau

class Trainer:
    def __init__(self, model_path, epochs=5000, patience=50, verbose=0, batch_size=32):
        self.epochs = epochs
        self.patience = patience
        self.model_path = model_path
        self.verbose = verbose
        self.batch_size = batch_size

    def train(self, model, X_train, y_train, X_validation, y_validation):
        datagen = ImageDataGenerator(
            featurewise_center=False,
            featurewise_std_normalization=False,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        datagen.fit(X_train)

        model_checkpoint = ModelCheckpoint(self.model_path, monitor='val_loss', verbose=self.verbose, save_best_only=True, mode='min', save_weights_only=False)
        early_checkpoint = EarlyStopping(monitor="val_loss", patience=self.patience, verbose=self.verbose)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(self.patience/4), verbose=self.verbose)

        callbacks = [model_checkpoint, History(), early_checkpoint, reduce_lr]
        return model.fit_generator(datagen.flow(X_train, y_train, batch_size=self.batch_size), 
                                    validation_data=(X_validation, y_validation),verbose=self.verbose,
                            epochs=self.epochs, steps_per_epoch=len(X_train) / self.batch_size,
                            callbacks=callbacks)