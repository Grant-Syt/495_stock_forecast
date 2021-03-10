import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from WindowGenerator import WindowGenerator

class ModelGenerator():

    def __init__(self, train_df, val_df, test_df, input_width, label_width, 
                shift, max_epochs, label_columns=None, patience=10):
        self.max_epochs = max_epochs
        self.patience = patience
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        conv_window = WindowGenerator(
            train_df=train_df, test_df=test_df, 
            val_df=val_df, input_width=input_width,
            label_width=label_width, shift=shift, 
            label_columns=label_columns)
        self.conv_window = conv_window

        conv_model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=32,
                                    kernel_size=(input_width,),
                                    activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1),])
        self.conv_model = conv_model

    def compile_and_fit(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.patience, mode='min')

        self.conv_model.compile(loss=tf.losses.MeanSquaredError(), 
            optimizer=tf.optimizers.Adam(), 
            metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.conv_model.fit(self.conv_window.train, 
            epochs=self.max_epochs, validation_data=self.conv_window.val, 
            callbacks=[early_stopping])
        return history

    def test_model(self):
        test_iterator = iter(self.conv_window.test)
        y_p = []
        y = []
        for inputs, labels in test_iterator:
            predictions = self.conv_model( inputs )
            for i in range( predictions.shape[0] ):
                y_p.append( predictions[i][0,:][0].numpy() )
                y.append( labels[i][0,:][0] )
        
        y_p = np.array( y_p )
        y = np.array( y )
        mae = mean_absolute_error( y_p, y )
        return y, y_p, mae

    def single_window(self, df):
        y_p = []
        conv_window = WindowGenerator(train_df=df, test_df=df, val_df=df,
                              input_width=self.input_width, label_width=self.label_width, 
                              shift=self.shift, label_columns=['y'])
        test_iterator = iter(conv_window.test)
        for inputs, labels in test_iterator:
            predictions = self.conv_model( inputs )
            for i in range( predictions.shape[0] ):
                y_p.append( predictions[i][0,:][0].numpy() )
        return y_p