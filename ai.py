import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import mnist, cifar10  # Örnek veri setleri
from sklearn.model_selection import train_test_split
import numpy as np

# Metin veri seti için örnek veri yükleme (MNIST kullanılabilir)
(x_train_text, y_train_text), (x_test_text, y_test_text) = mnist.load_data()
x_train_text, x_test_text = x_train_text / 255.0, x_test_text / 255.0  # Normalize etme

# Metin veri seti için örnek veri yükleme (CIFAR-10 kullanılabilir)
(x_train_image, y_train_image), (x_test_image, y_test_image) = cifar10.load_data()
x_train_image, x_test_image = x_train_image / 255.0, x_test_image / 255.0  # Normalize etme

# Metin verisini işleme
tokenizer = Tokenizer(num_words=10000)  # 10,000 en sık kullanılan kelimeye sınırlama
tokenizer.fit_on_texts([str(x) for x in x_train_text])
sequences_train = tokenizer.texts_to_sequences([str(x) for x in x_train_text])
sequences_test = tokenizer.texts_to_sequences([str(x) for x in x_test_text])
x_train_text = pad_sequences(sequences_train, maxlen=100, padding='post')
x_test_text = pad_sequences(sequences_test, maxlen=100, padding='post')

# Metin verisi için model oluşturma (örnek bir LSTM modeli)
model_text = tf.keras.Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(64),
    Dense(10, activation='softmax')
])

# Görüntü verisi için model oluşturma (örnek bir CNN modeli)
model_image = tf.keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Derin öğrenme modeli oluşturma (örnek bir GAN modeli)
generator = tf.keras.Sequential([
    Dense(128, activation='relu', input_shape=(100,)),
    Dense(784, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28))
])

discriminator = tf.keras.Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

gan = tf.keras.Sequential([generator, discriminator])

# Verileri eğitim ve test için ayırma
x_train_text, x_val_text, y_train_text, y_val_text = train_test_split(x_train_text, y_train_text, test_size=0.2, random_state=42)
x_train_image, x_val_image, y_train_image, y_val_image = train_test_split(x_train_image, y_train_image, test_size=0.2, random_state=42)

# Metin modelini derleme ve eğitme
model_text.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_text.fit(x_train_text, y_train_text, epochs=5, validation_data=(x_val_text, y_val_text))

# Görüntü modelini derleme ve eğitme
model_image.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model_image.fit(x_train_image, y_train_image, epochs=5, validation_data=(x_val_image, y_val_image))

# Derin öğrenme modelini derleme ve eğitme
gan.compile(optimizer='adam', loss='binary_crossentropy')
gan.fit(x_train_text, y_train_text, epochs=5)