import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed, Embedding, Bidirectional
# Import CRF from a compatible source or use an alternative implementation
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as sklearn_f1_score
from seqeval.metrics import precision_score, recall_score, f1_score as seqeval_f1_score, classification_report
from sklearn_crfsuite.metrics import flat_classification_report
import matplotlib.pyplot as plt
from cls_sentence import sentence  # Assuming this is a custom class you've defined elsewhere.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def load_data(filename='label_ner.csv'):
    df = pd.read_csv(filename, encoding='utf-8-sig')
    df = df.ffill()
    return df

def process_data(df):
    # Create a mapping for words to index
    words = list(set(df["Word"].values))
    words.append("PAD")
    words.append("UNK")
    word2idx = {w: i for i, w in enumerate(words)}

    # Create a mapping for tags to index
    tags = list(set(df["Tag"].values))
    tag2idx = {t: i for i, t in enumerate(tags)}

    # Group data by sentence and create sequences
    agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(), s["Tag"].values.tolist())]
    grouped = df.groupby("Sentence #").apply(agg_func)
    sentences = [s for s in grouped]

    # Convert words and tags to their indices
    X = [[word2idx[w[0]] if w[0] in word2idx else word2idx["UNK"] for w in s] for s in sentences]
    y = [[tag2idx[t[1]] for t in s] for s in sentences]

    # Pad sequences for equal length input
    max_len = 60  # or another appropriate length
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word2idx["PAD"])
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["PAD"])

    # Convert label sequences to categorical, for each label
    y = [to_categorical(i, num_classes=len(tags)) for i in y]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

    return X_train, X_test, y_train, y_test, word2idx, tag2idx, tags


def build_model(num_tags, words, max_len=60, embedding_dim=40):
    input_layer = Input(shape=(max_len,))
    embedding_layer = Embedding(input_dim=len(words) + 2, output_dim=embedding_dim, mask_zero=True)(input_layer)
    bi_lstm = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(embedding_layer)
    time_distributed = TimeDistributed(Dense(num_tags, activation='softmax'))(bi_lstm)  # Adjust the number of units to num_tags
    
    model = Model(inputs=input_layer, outputs=time_distributed)
    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

df = load_data()
X_train, X_test, y_train, y_test, word2idx, tag2idx, tags = process_data(df)
num_tags = len(tags)  # Number of unique tags

if not os.path.exists("model.keras"):
    print("Data not found, make it!")
    df = load_data()
    X_train, X_test, y_train, y_test, word2idx, tag2idx, idx2word, idx2tag, words, tags = process_data(df)
    num_tags = len(tags)  # Make sure this reflects the correct number of unique tags
    model = build_model(num_tags, words, max_len=60, embedding_dim=40)
    checkpoint = ModelCheckpoint('model.keras', verbose=0, save_best_only=True, monitor='val_loss')
    history = model.fit(X_train, np.array(y_train), batch_size=64, epochs=50, validation_split=0.1, callbacks=[checkpoint])
else:
    df = load_data()
    _, _, _, _, _, _, _, _, words, tags = process_data(df)
    num_tags = len(tags)
    model = build_model(num_tags, words)
    model.load_weights("model.keras")