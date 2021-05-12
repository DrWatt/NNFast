#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 15:49:30 2021

@author: Dr. Marco Lorusso
"""

import time
import argparse
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import requests
from joblib import load, dump

from subprocess import check_output
#numpy
import numpy as np
# pandas
import pandas as pd
# matplotlib
import matplotlib.pyplot as plt
# TensorFlow
import tensorflow as tf
# QKeras
from qkeras.utils import load_qmodel
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu,smooth_sigmoid,quantized_tanh

from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from pickle import dump,load

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"  


seed = 1563
np.random.seed(seed)
encoder = LabelEncoder()
scale =MinMaxScaler()
curdir = os.getcwd()
Nfolder = check_output("n=0; while [ -d \""+curdir+"/out_${n}\" ]; do n=$(($n+1)); done; mkdir \""+curdir+"/out_${n}\"; echo $n",shell=True)
fold = curdir+"/out_"+str(int(Nfolder))


def baseline_model(indim=7,hidden_nodes=[8,8],jsonpath=None,outdim=1,Quant=False,multiclass=False):
    '''
    Model constructor definition, as needed to use scikit-learn wrapper with keras.    
    
    Parameters
    ----------
    indim : int, optional
        Number of features of dataset and dimension of input layer. The default is 7.
    hidden_nodes : list, optional
        List of number of nodes per layer. The default is [8,8].
    outdim : int, optional
        Number of classes and dimension of output layer. The default is 9.

    Returns
    -------
    model : keras.engine.sequential.Sequential
        Sequntial NN object to be used inside KerasClassifer method.

    '''
    model = tf.keras.Sequential()
    
    #Nodes of the NN.
    try:
        layoutdict = json.load(open(jsonpath))
    except Exception:
        print("Json file with NN layout not found or not provided, using default lay.json")
        layoutdict = json.load(open("lay.json"))
    if Quant:
        model.add(QDense(hidden_nodes[0],  input_shape=(indim,),kernel_quantizer=quantized_bits(16,1),bias_quantizer=quantized_bits(16,1), kernel_initializer='random_normal'))
        model.add(QActivation(activation=quantized_relu(16,1), name='relu1'))
        for a in hidden_nodes[1:]:
            model.add(QDense(a,kernel_quantizer=quantized_bits(16,1),bias_quantizer=quantized_bits(16,1)))
            model.add(QActivation(activation=quantized_relu(16,1), name='relu'))
        model.add(QDense(outdim, kernel_quantizer=quantized_bits(16,1),bias_quantizer=quantized_bits(16,1)))
        model.add(QActivation(activation=quantized_relu(16,1), name='relufin'))
    else:
        model.add(eval("tf.keras.layers."+layoutdict["Input"]["type"])(units=layoutdict["Input"]["nodes"],input_dim=indim,activation=layoutdict["Input"]["activation"]))
        #model.add(tf.keras.layers.Dense(hidden_nodes[0], input_dim=indim, activation='relu'))    
        hidden = layoutdict["Hidden Layers"]
        
        for lays in hidden:
            
            model.add(eval("tf.keras.layers."+lays['type'])(units = lays["nodes"],activation=lays["activation"]))
    
        
        # for a in hidden_nodes[1:]:
        #     model.add(tf.keras.layers.Dense(a,activation='relu'))
    
        model.add(eval("tf.keras.layers."+layoutdict["Output"]["type"])(outdim, activation=layoutdict["Output"]["activation"]))
        #model.add(tf.keras.layers.Dense(outdim, activation='relu'))
    # Compile model.
    if multiclass:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def model_upload(modpath,Quant=False):
    '''
    Function to load pretrained NN.

    Parameters
    ----------
    modpath : String
        path (local or URL) of model in joblib format..

    Returns
    -------
    keras.wrappers.scikit_learn.KerasClassifier 
        Wrapper from the Scikit.learn library of a the Keras Classifier.
    or
    xgboost.core.Booster
        Booster is the model of xgboost, that contains low level routines for training, prediction and evaluation.

    '''
    if("http" in modpath):
        print("Downloading Model")
        try:    
            # Download
            mod = requests.get(modpath)
            mod.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            raise 
        
        # Writing model on disk.
        if (".joblib" in modpath):
            with open(fold+"/model.joblib","wb") as o:
                o.write(mod.content)
            modpath = fold+"/model.joblib"
        if (".h5" in modpath):
            with open(fold+"/model.h5","wb") as o:
                o.write(mod.content)
            modpath = fold+"/model.h5"
    print("Loading Model from Disk")

    # Uploading model from disk. 
    try:
        if Quant:
            estimator = load_qmodel(modpath)
        else:
            estimator = tf.keras.models.load_model(modpath)
    except:
        print("Error: Could not load model")
        raise
    return estimator

def data_upload(datapath,name="dataset"):
    '''
    Function to load data from disk or using an URL.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.

    Returns
    -------
    pandas.dataframe
        Dataframe containing data used for training and/or inference.

    '''
    if("http" in datapath):
        print("Downloading Dataset")
        try:
            # Download
            dataset = requests.get(datapath)
            dataset.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            raise        
        # Writing dataset on disk.    
        with open(fold+"/" + name + ".csv","wb") as o:
            o.write(dataset.content)
        datapath = fold+"/" + name + ".csv"
    print("Loading Dataset from Disk")
    
    # Reading dataset and creating pandas.DataFrame.
    dataset = pd.read_csv(datapath,header=0)
    print("Entries ", len(dataset))        
    
    return dataset

def correlation_matrix(data):
    '''
    Quick correlation matrix printer

    Parameters
    ----------
    data : pandas.dataframe
        Dataframe containing data.

    Returns
    -------
    pandas.dataframe
        Dataframe rescaled using min-max normalization.

    '''
    

    f = plt.figure(figsize=(19, 15))
    plt.matshow(data.corr(), fignum=f.number)
    plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
    plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
    plt.show()
    plt.savefig(fold + "/Correlation_training_set.png")
    plt.clf()
    
   

    return f


def predictor(datapath,modelpath,performance=False,NSamples=0,resultlabel=False,Quant=False):
    '''
    Function to compute classification of a dataset using a pretrained NN.
    

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format..
    modelpath : String
        path (local or URL) of model in joblib format..
    performance : Boolean, optional
        Set between two return mode: False -> return only predictions; True -> return predictions and true labels if provided (for evaluating performance). The default is False.
    NSamples : int, optional
        number of entries used of the dataset. If NSamples == 0 or NSamples > data size the all dataset will be used. The default is 0.

    Returns
    -------
    pandas.dataframe
        Dataframe containing inferences made by the model for each entry of the data in input.
        
    list of pandas.dataframe
        List made up of two dataframes, the first contains the inferences, the second contains the true labels for validation.

    '''
    # Loading dataset and preprocessing it.

    dataset = data_upload(datapath)
    # Loading NN.
    estimator = model_upload(modelpath,Quant)
    if resultlabel==-1 or resultlabel==False: 
            resultlabel=dataset.columns[-1]
            print("Target value not specified: using last column")
    # Failed loading handling.
    # if estimator == 404:
    #     return 404
    if type(estimator) != tf.keras.models.Sequential:
        print("Check loaded model compatibility.")
        raise TypeError(estimator)
    
    if resultlabel in dataset.columns: dataset = dataset.drop(columns=[resultlabel])
    # Handling of number of entries argument (NSample).
    if NSamples == 0:
        X = dataset
        
    elif NSamples > dataset.size:
        print("Sample requested is greater than dataset provided: using whole dataset")
        X = dataset
    else:
         X = dataset.head(NSamples)
    print("Thinking about the results")
    # Actual prediction method + inverse encoding to get actual BX values.
    pred=estimator.predict(X)

    # condition to return also labels.
    if performance:
        try:
            labels = dataset.head(len(X.index)).get(resultlabel)
        except:
            print("Error: Could not find label of the predicted quantity")
            raise
        return [pred,labels]

    return pred


def plotting_NN(estimator,history):
    '''
    Plotting function that saves three different .png images: 
    1) Representation of the neural network;
    2) Plot of the model accuracy thorugh epochs for training and validation sets;
    3) Plot of the model loss function thorugh epochs for training and validation sets.

    Parameters
    ----------
    estimator : keras.wrappers.scikit_learn.KerasClassifier
        Object containing NN model.
    history : keras.callbacks.History
        Return of fit function of the NN model.

    Returns
    -------
    None.

    '''
    #plot_model(estimator.model, to_file='model.png',show_shapes=True)
    
    # Accuracy and Loss function plots saved in png format.
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')      
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(fold+"/Keras_NN_Accuracy.png")
        plt.clf()
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(fold+"/Keras_NN_Loss.png")

def data_encoder(datapath,targetlabel,NSample=None):
    '''
    Function performing one-hot encoding.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.
    NSample : int, optional
        number of entries used of the dataset. If NSamples == None or NSamples > data size the all dataset will be used.. The default is None.

    Returns
    -------
    List of pandas.dataframe
        List made up of two Dataframes: the first contains the preprocessed data and the second one contains the one hot encoded labels.

    '''
    print("encoding\\\\\\\\\\\/")
    # Uploading preprocessed dataset.
    dataset = data_upload(datapath)

    
    if targetlabel==-1: 
        targetlabel=dataset.columns[-1]
        print("Target value not specified: using last column")

    # Handling of number of entries argument (NSample).
    if NSample == None or NSample == 0:
        X = dataset.drop(columns=[targetlabel]).to_numpy()
        target = dataset[targetlabel].to_numpy()
    elif NSample > dataset.size:
        print("Sample requested is greater than dataset provided: using whole dataset")
        X = dataset.drop(columns=[targetlabel]).to_numpy()
        target = dataset[targetlabel].to_numpy()
    else:
        X = dataset.drop(columns=[targetlabel]).to_numpy()
        target = dataset[targetlabel].to_numpy()
    
    # Encoding BXs to have labels from 0 to 9.
    #encoder = LabelEncoder()
    encoder.fit(target)
    dump(encoder,open("encoder.pkl","wb"))
    encoded_target = encoder.transform(target)
    transformed_target = tf.keras.utils.to_categorical(encoded_target)

    return [X,transformed_target]  
  
def training_model(datapath,targetname, NSample=0, par = [2,30,0.3],plotting=False,multiclassification=False,Quant=False):
    '''
    NN training function.

    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.
    targetname : String
        Label of the column containing the training target true values.
    NSample : int, optional
        number of entries used of the dataset. If NSamples == 0 or NSamples > data size the all dataset will be used.. The default is 0.
    par : List of int,int,float, optional
        list of paramaters passed to the NN costructor [number of epochs the NN will be trained for, size of the batches used to update the weights, fraction of the input dataset used for validation]. The default is [48,30,0.3].

    Returns
    -------
    pandas.DataFrame
        Values assumed by evaluation metrics through the epochs.

    '''
    
    # Loading and preparing data for training.

    # Setting default values in case of some missing parameter.
    if par[0] == 0 : par[0] = 48
    if par[1] == 0 : par[1] = 30
    if par[2] == 0 : par[2] = 0.3
    outdim=1
    print(par)
    
    if multiclassification:
        dataset,truelabel = data_encoder(datapath,targetname)
        outdim = truelabel.shape[1]

    else:
        dataset = data_upload(datapath)
        if targetname==-1: 
            targetname=dataset.columns[-1]
            print("Target value not specified: using last column")
        try:
            truelabel = dataset[targetname].to_numpy().reshape(dataset.shape[0],1)
        except KeyError:
            print("Error: target not provided or not found in dataset")
            raise
        truelabel=scale.fit_transform(truelabel)
        dataset = dataset.drop(columns=[targetname]).to_numpy()
    
    # Model constructor
    estimator = baseline_model(indim=dataset.shape[1],outdim=outdim,Quant=Quant,multiclass=multiclassification)
    
    # Training method for our model. 
    history = estimator.fit(dataset, truelabel, epochs=par[0], batch_size=par[1],verbose=1,validation_split=par[2])

    # Saving trained model on disk. (Only default namefile ATM)
    estimator.save(fold+"/KerasNN2_Model.h5")
    if plotting:
        plotting_NN(estimator, history)
    # Returning values assumed by evaluation metrics through the epochs.
    return pd.DataFrame.from_dict(history.history)




def run(argss):
    '''
    Main function invoked by execution in shell.

    Parameters
    ----------
    argss : argparse.ArgumentParser
        Arguments parsed to the invoked function. Contains flags which control the execution of the script: e.g. the model of choice and if you want to train or infer.

    Returns
    -------
    Dictionary
        Values assumed by evaluation metrics through the epochs for both models.

    '''
    resul = {'KerasNN':0,'QKerasNN':0}
    argss.nn = True
    # if argss.nn == 0  and argss.hls == 0:
    #     os.system("rm -r " + fold)
    #     raise Exception("Choose a model using the --xgb and/or --nn flags or use the --hls flag to activate hls4ml functionality.")
    #     #print("Choose a model using the --xgb and/or --nn flags")
    if argss.targetvar==None: argss.targetvar=-1
    if argss.data==None: argss.data = "https://raw.githubusercontent.com/DrWatt/softcomp/master/datatree.csv"
    # Routine followed when --nn is True
    if argss.nn:
        
        # Selection between prediction, using a pretrained model, and training a new one.
        if argss.modelupload:
            if (argss.Qnn): 
                print("Inference using a Quantized NN")
                pred = predictor(argss.data,argss.modelupload,Quant=True)
            else:
                pred = predictor(argss.data,argss.modelupload,resultlabel=argss.targetvar)
            pred.astype(float).tofile(fold+"/kerasres.csv",sep='\n',format='%f')
            print("Predictions saved in .csv format")
        else:
            # try:
            #     nnparams = json.load(open(pars.nnparams)) if pars.nnparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.nnparams))
            # except Exception:
            #     print("NN parameters not found! Using default values")
            #     nnparams = [48,30,0.3]
            
            # Reading list of parameters. If no parameters are specified, default values will be used.
            if argss.nnparams==None:argss.nnparams = [0,0,0]
            print (argss.nnparams)
            pr = [int(argss.nnparams[0]),int(argss.nnparams[1]),float(argss.nnparams[2])]
            print(pr)
            
            if (argss.Qnn): 
                print("Training a Quantized NN")
                model = training_model(argss.data,argss.targetvar,
                            par=pr,
                            plotting=True,multiclassification=argss.C,Quant=True)
            # Construction and training of Keras NN.
            else:
                model = training_model(argss.data,argss.targetvar,
                                par=pr,
                                plotting=True,multiclassification=argss.C)
            
            # results = 1- nn_performance(model,"datatree.csv")
            # print("Neural Network's accuracy: ", results)
            print("Plots of evaluation metrics vs epochs saved. \nModel in .h5 format saved for prediction and testing")
            resul['KerasNN'] = model
    #print("XGboost's accuracy", bdtres)
    # elif argss.hls:
    #     if argss.modelupload:
    #         model = model_upload(argss.modelupload)
    #     else:
    #         raise Exception("No keras model provided.")
            
    #     config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    #     print(config)

    #     hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, output_dir=fold+'/model_1/hls4ml_prj',fpga_part='xc7k70t-fbv676-1')

    #     hls_model.compile()
    #     hls_model.build()
    
    return resul


if __name__ == '__main__':
    time0 = time.time()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    tf.autograph.set_verbosity(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=str,help="Url or path of dataset in csv format.")
    parser.add_argument('--nn', action='store_true', help='If flagged activate keras nn model')
    parser.add_argument('--Qnn', action='store_true', help='If flagged activate Qkeras nn model')
    parser.add_argument('-C', action='store_true', help='If flagged build a NN for classification')
    #parser.add_argument('--hls', action='store_true', help='If flagged activate parsing of keras model to HDL')
    parser.add_argument('--nnlayout', type=str, help="Layout for the Keras NN in JSON format")
    parser.add_argument("--targetvar",type=str,help="Target value for training or inference")
    parser.add_argument('--nnparams',nargs='+', help="Hyperparameters for Keras NN")
    #parser.add_argument('-p', action='store_true', help='If flagged set predecting mode using a previously trained model')
    parser.add_argument('--modelupload',type=str,help="Url or path of model in joblib format")
    parser.add_argument('--nogpu',action='store_true',help="Deactivate training on GPU")
    #parser.set_defaults
    #print(parser.parse_args())
    pars = parser.parse_args()
    #xgparams = json.load(open(pars.xgparams)) if pars.xgparams[0][0] == '/' else json.load(open(os.path.dirname(os.path.realpath(__file__))+'/'+pars.xgparams))

    if pars.nogpu: os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
    run(pars)
    print("Files saved in "+ fold)
    print("Executed in %s s" % (time.time() - time0))
    
