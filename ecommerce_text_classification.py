#%%
#1. Import packages
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import LabelEncoder
from review_handler import remove_unwanted_strings

# %%
#2. Data Loading
PATH = os.getcwd()
CSV = os.path.join(PATH, 'ecommerceDataset.csv')
column_names = ['category', 'text']
df = pd.read_csv(CSV, header=None)
df.columns = column_names
# %%
#3. Data inspection
print(df.info())
print("-"*20)
print(df.describe())
print("-"*20)
print(df.isna().sum())
print("-"*20)
print(df.duplicated().sum())

# %%
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(df.info())

# %%
#4. The review is the feature, the sentiment is the label
features = df['text'].values
labels = df['category'].values

# %%
#5. Convert label into integers using LabelEncoder
label_encoder = LabelEncoder()
label_processed = label_encoder.fit_transform(labels)

# %%
#6. Data preprocessing
#(A) Remove unwanted strings from the data
features_removed = remove_unwanted_strings(features)

# %%
#7. Define some hyperparameters
vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = 0.3

#%%
#8. Perform train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features_removed,label_processed,train_size=training_portion,random_state=42)

# %%
#9. Perform tokenization
from tensorflow import keras

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size,split=" ",oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)

# %%
word_index = tokenizer.word_index
print(dict(list(word_index.items())[0:10]))

# %%
X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# %%
#10. Perform padding and truncating
X_train_padded = keras.preprocessing.sequence.pad_sequences(X_train_tokens,maxlen=max_length,padding=padding_type,truncating=trunc_type)
X_test_padded = keras.preprocessing.sequence.pad_sequences(X_test_tokens,maxlen=max_length,padding=padding_type,truncating=trunc_type)

# %%
#11. Model development
#(A) Create the sequential model
model = keras.Sequential()

#(B) Create the input layer, in this case, it can be the embedding layer
model.add(keras.layers.Embedding(vocab_size,embedding_dim))
#(B) Create the biderictional LSTM layer
model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim)))
#(C) Classsification layers
model.add(keras.layers.Dense(embedding_dim,activation='relu'))
model.add(keras.layers.Dense(len(np.unique(y_train)), activation='softmax'))

model.summary()

#%%
#Create a metric using keras backend for f1_score
def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score

# %%
#12. Model compilation
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy', f1_score])

#Create a TensorBoard callback object for the usage of TensorBoard
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# %%
#13. Model Training
history = model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=10, batch_size=64, callbacks=tensorboard_callback)

# %%
print(history.history.keys())

# %%
#Plot accuracy graphs
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(["Train accuracy","Test accuracy"])
plt.show()

# %%
#14. Model deployment
test_string=(["Paper Plane Design Framed Wall Hanging Motivational Office Decor Art Prints (8.7 X 8.7 inch) - Set of 4 Painting made up in synthetic frame with uv textured print which gives multi effects and attracts towards it.",
               "History of English Literature Book",
               "Symbol Men's Regular Fit Shorts",
               "Lenovo Tab4 8 Plus Tablet Aurora Black"])

#%%
test_string_removed = remove_unwanted_strings(test_string)

#%%
test_string_tokens = tokenizer.texts_to_sequences(test_string_removed)

#%%
test_string_padded = keras.preprocessing.sequence.pad_sequences(test_string_tokens,maxlen=max_length)

# %%
y_pred = np.argmax(model.predict(test_string_padded),axis=1)

# %%
label_map = ['Household','Books','Clothing & Accesories','Electronic']
predicted_sentiment = [label_map[i] for i in y_pred]

print(predicted_sentiment)
# %%
#16. Save model and tokenizer
import os

PATH = os.getcwd()
print(PATH)

# %%
# Model save
model_save_path = os.path.join(PATH, 'save_model', 'sentiment_analysis_model.h5')
model.save(model_save_path)

# %%
#tokenizer save path
import pickle

tokenizer_save_path = os.path.join(PATH, 'save_model', "tokenizer.json")
with open(tokenizer_save_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# %%