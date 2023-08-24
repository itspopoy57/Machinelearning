import tensorflow as tf

def MSE(y_true , y_pred):
  return tf.metrics.mean_squared_error(y_true = y_test,
                                        y_pred = tf.squeeze(xamp1P));
#making function to calculate the MAE and MSE
def MAE(y_true , y_pred):
  return tf.metrics.mean_absolute_error(y_true = y_test,
                                        y_pred = tf.squeeze(xamp1P));
