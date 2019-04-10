from trainer import Trainer
from cnn_model import CNNModel
from fer_reader import FERReader
import keras

if __name__ == "__main__":
    reader = FERReader("../datasets/fer2013.csv", "../datasets/fer2013new.csv")
    X_train, y_train, X_validation, y_validation, X_test, y_test = reader.read()

    model = CNNModel((48, 48, 1), reader.emotion_count).build_model()
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer="adam", metrics=['acc'])

    trainer = Trainer("../model/model3_weights.{epoch:02d}-{val_loss:.4f}_{val_acc:.4f}.hdf5")
    trainer.train(model, X_train, y_train, X_validation, y_validation)

