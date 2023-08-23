def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = xamp1P
                     ):
  """
  Plots Training data, test Data and compares the predictions to ground truth
  """

  plt.figure(figsize=(10,7))
  #plot here
  plt.scatter(train_data, train_labels, c="b", label="Training data")
  #plot the other on green for the prediction or test
  plt.scatter(test_data, test_labels, c="g", label="Test Data")
  # plot the models predictions in red
  plt.scatter(test_data, predictions, c="r", label = "predictions")
  plt.legend();

