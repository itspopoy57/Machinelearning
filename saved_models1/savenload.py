saving them allow us to use them outside of google colab or whatever they we trained for? ,

Ways to save the models

The savemodel format
The HDF5 format

xamp1.save("almost_perfect_prediction_xamp1")
xamp1.save("model_xamp1.h5")

xamp1_load = tf.keras.models.load_model("/content/almost_perfect_prediction_xamp1")
#also you have to save this to google drive coz after reconnect all the files saved are all temp/
xamp1_load.summary(), xamp1.summary()


xamp1_load_pred = xamp1_load.predict(X_test)
xamp1_prediction2 = xamp1.predict(X_test)


xamp1_load_pred == xamp1_prediction2

#download a file from google colab
from google.colab import files 
files.download("/content/almost_perfect_prediction_xamp1")
