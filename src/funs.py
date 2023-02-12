import matplotlib.pyplot as plt

def plot_history_plus(history):
  plt.plot(history['loss'], label='Training loss')
  plt.plot(history['val_loss'], label='Validation loss')
  plt.legend()
  plt.show()

  plt.plot(history['accuracy'], label='Training accuracy')
  plt.plot(history['val_accuracy'], label='Validation accuracy')
  plt.legend()