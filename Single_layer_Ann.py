# -- coding: utf-8 --
"""
Created on Fri May 01 09:20:35 2020

@author: Suman Bhurtel
"""

import zipfile 
import os
import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing


#Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

my_zip = zipfile.ZipFile('C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang.zip') # Specify your zip file's name here
for file in my_zip.namelist():
    # if my_zip.getinfo(file).filename.endswith('.txt'):
        my_zip.extractall() # extract the file to current folder if it is a text file

        
# define a function that creates a data frame of the all file counting number of occurances
def matrix_creator(read_files):
    np_array_values = []
    
    
    for files in read_files:
        data1 = pd.read_csv(files,sep=':,!',header= None)
        np_array_values.append(data1)
    
    
    
    charcount = {} 
    charcount_ini = {}
    #dictionary to hold char counts
    validchars = "abcdefghijklmnopqrstuvwxyz" # only these counted
     
    print(": Letter : Frequency :")
     
    for i in range(97,123): # lowercase range
        b= (chr(i)) # the chars a-z
        charcount[b] = 0 # initialize count
    for i in range(97,123): # lowercase range
        c = (chr(i)) # the chars a-z
        charcount_ini[c] = 0 # initialize count
    
    #print(charcount)
    dataframe = pd.DataFrame(charcount_ini,index=['row_i'])
        
    length = len(np_array_values)
    for j in range(length):  
        data123 = np_array_values[j]
        
        for k in data123.index:
            line = data123[0][k]
            words = line.split(" ")
            for word in words: 
                chars = list(word)
                for c in chars:
                    if c.isalpha():
                        if c.isupper():
                            c = c.lower()
                        if c in validchars: 
                            charcount_ini[c] += 1
                
        #dataframe.append(pd.DataFrame(charcount),index=['2'])
        dataframe_i = pd.DataFrame(charcount_ini,index=['row_i'])
         
         #print(dataframe_i)
         #dataframe.append(dataframe_i)
        dataframe = dataframe.append(dataframe_i) 
        charcount_ini = charcount
    return dataframe


# create data frame of english data

read_files_english = glob.glob(os.path.join("C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang/English","*.txt"))
training_data_English = matrix_creator(read_files_english)
training_data_English = training_data_English.assign(true_languase=(1,1,1,1,1,1,1,1,1,1,1)) #labeling the data
training_data_English = training_data_English.iloc[1:, 0:27].values


# crete data frame for german data

read_files_german = glob.glob(os.path.join("C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang/German","*.txt"))
training_data_German = matrix_creator(read_files_german)
training_data_German = training_data_German.assign(true_languase = (0,0,0,0,0,0,0,0,0,0,0))
training_data_German = training_data_German.iloc[1:, 0:27].values


# create data frame of polish
read_files_polish = glob.glob(os.path.join("C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang/Polish","*.txt"))
training_data_Polish = matrix_creator(read_files_polish)
training_data_Polish = training_data_Polish.assign(true_languase=(2,2,2,2,2,2,2,2,2,2,2))
training_data_Polish = training_data_Polish.iloc[1:, 0:27].values

training_data_final = np.concatenate((training_data_English,training_data_Polish,training_data_German),axis=0)

#one hot encoding


X_train = training_data_final[1:,0:26]
# normalize the x_train
X_train = preprocessing.normalize(X_train, norm='l2')

y_train = training_data_final[1:,26]
y_train = keras.utils.to_categorical(y_train, num_classes=3)


####### Part 2 - let's make the ANN! ###########

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer as it is Single Layer ANN with input vector with 26 Latin alphabets
# using uniform weight distribution and assigning 6 nodes in the first hidden layer.
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 26)) 



# Adding the output layer
classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'sigmoid')) #using Sigmoid Activation function

# Compiling the ANN
#back propagation(optimizing) using Adam.
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) 

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 3, epochs = 1000)

################# Part 3 - Making predictions and evaluating the model ##########


# reading the input data to make a model
read_files_test_english = glob.glob(os.path.join("C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang.test/English","*.txt"))
read_files_test_polish = glob.glob(os.path.join("C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang.test/Polish","*.txt"))
read_files_test_german = glob.glob(os.path.join("C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/lang.test/German","*.txt"))
test_data_german = matrix_creator(read_files_test_german)
test_data_english = matrix_creator(read_files_test_english)
test_data_polish = matrix_creator(read_files_test_polish)

# adding the true languase column at the end to train tha data in supervised way
test_data_polish = test_data_polish.assign(true_languase = (2,2,2,2))
test_data_polish = test_data_polish.iloc[1:, 0:27].values
test_data_english = test_data_english.assign(true_languase=(1,1,1,1))
test_data_english = test_data_english.iloc[1:,0:27].values
test_data_german = test_data_german.assign(true_languase=(0,0,0,0))
test_data_german = test_data_german.iloc[1:,0:27].values


# combining the datas to make one single data 
test_data_final = np.concatenate((test_data_english,test_data_polish,test_data_german),axis=0)



#one hot encoding the test data y value
X_test = test_data_final[0:, 0:26]
y_test = test_data_final[0:,26]
y_test = keras.utils.to_categorical(y_test, num_classes=3)

# # Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred =(y_pred>0.5).astype(int)


# creating the confusion matrix 
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in y_pred ]

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)

# take input from user 
#C:/Users/oookr/Desktop/OneDrive_1_4-28-2020/user_input_to_check(example text file)/English
user_input = input(str('Please provide the path to the file that you want programm to predict: '))
read_user_input = glob.glob(os.path.join(user_input,"*.txt"))
user_input_to_matrix = matrix_creator(read_user_input)
user_input_to_matrix = user_input_to_matrix.iloc[1:,0:26].values
user_input_to_matrix= preprocessing.normalize(user_input_to_matrix, norm='l2')
# predict the user input
predict_user_text = classifier.predict(user_input_to_matrix)

# provide the predicted out put to the user input
predict_user_text = (predict_user_text > 0.5).astype(int)
if (predict_user_text == [[0,1,0]]).all():
    print('your given text is in English language')
elif (predict_user_text == [[0,0,1]]).all():
    print('your given text is in Polish language')
elif (predict_user_text == [[1,0,0]]).all():
    print('your given text is in German language')
else:
    print('sory could not predict may be I am a stupid Machine!! :( ')