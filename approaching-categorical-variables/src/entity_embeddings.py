import os
import gc
from venv import create
import joblib
import pandas as pd
import numpy as np

from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils

def create_model(data, catcols):
    """
    This function returns a compiled tf.keras model
    for entity embeddings.
    :param data: this is a pandas dataframe
    :param catcols: list of categorical column names
    :return: compiled tf.keras model
    """

    # initialize list of inputs for embeddings
    inputs = []

    # initialize list of outputs for embeddings
    outputs = []

    print("I entered the function")
    # loop over all categorical columns
    for i, c in enumerate(catcols):

        # find the number of unique values in the column
        num_unique_values = int(data[c].nunique())
        print(f"Column {i} has {num_unique_values} unique values")
        # simple dimension of embedding calculator.
        # min size is half of the number of unique values;
        # max size is 50. max size depends on the number of
        # unique categories too. 50 is quite sufficient most
        # of the times but if you have millions of unique 
        # values, you might need a larger dimension
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=c)(inp)

        # 1-d spatial dropout is the standard for embedding layers;
        # it can be used in NLP tasks too
        out = layers.SpatialDropout1D(0.3)(out)

        # reshape the input to the dimension of embedding. this becomes
        # our output layer for current features
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        # add input to input list
        inputs.append(inp)

        # add output to output list
        outputs.append(out)

    # concatenate all output layers
    x = layers.Concatenate()(outputs)

    # add a batchnorm layer.
    # from here, everything is up to you;
    # you can try different architectures;
    # this is the architecture I like to use;
    # if you have numerical features, you
    # should add them here or in concatenate layer
    x = layers.BatchNormalization()(x)

    # a bunch of dense layers with dropout.
    # start with 1 or 2 layers only
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    # using softmax and treating it as a two-class problem;
    # you can also use sigmoid, then you need to use only one output class
    y = layers.Dense(2, activation="softmax")(x)

    # create final model
    model = Model(inputs=inputs, outputs=y)

    # compile the model; we use adam and binary cross entropy.
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model

def run(fold):
    #load the full training data with folds
    df = pd.read_csv("../input/cat_train_folds.csv")

    # all columns are features except id, target, and kfold columns
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    # fill all NaN values with NONE
    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    # encode all features with Label Encoder individually;
    # in a live setting, label encoders should be saved
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df[feat] = lbl_enc.fit_transform(df[feat].values)

    # get training and validation data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # create tf.keras model
    model = create_model(df, features)

    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    # fetch target columns
    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # convert target columns to categories
    # this is just binarization
    ytrain_cat = utils.to_categorical(ytrain)
    y_valid_cat = utils.to_categorical(yvalid)

    model.fit(xtrain,
              ytrain_cat,
              validation_data=(xvalid, y_valid_cat),
              verbose=1,
              batch_size=1025,
              epochs=3)

    # generate validation predictions
    valid_preds = model.predict(xvalid)[:, 1]

    # print roc auc score
    print(metrics.roc_auc_score(yvalid, valid_preds))

    K.clear_session()

if __name__ == "__main__":
    run(0)
