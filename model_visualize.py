from keras.models import load_model
model = load_model('model.h5')

# keras 2.0.9 syntax
from keras.utils import plot_model
plot_model(model, to_file='writeup_report/model.png', show_shapes=True)
