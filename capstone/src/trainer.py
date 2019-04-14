from fer_generator import FERGenerator
from keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau

class Trainer:
    def __init__(self, model_path, epochs=5000, patience=100, verbose=0, batch_size=32):
        self.epochs = epochs
        self.patience = patience
        self.model_path = model_path
        self.verbose = verbose
        self.batch_size = batch_size

    def train(self, model, fer_reader, X_train, X_validation):
        train_generator = FERGenerator(fer_reader, X_train, fer_reader.train_set, self.batch_size)
        validation_generator = FERGenerator(fer_reader, X_validation, fer_reader.validation_set, self.batch_size)

        model_checkpoint = ModelCheckpoint(self.model_path, verbose=self.verbose, save_best_only=True)
        early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=int(self.patience/5), verbose=self.verbose)

        callbacks = [model_checkpoint, History(), early_stopping, reduce_lr]
        return model.fit_generator(train_generator, validation_data=validation_generator, verbose=self.verbose,
                            epochs=self.epochs, steps_per_epoch=len(X_train) / self.batch_size, callbacks=callbacks)