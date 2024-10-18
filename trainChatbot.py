#import codes
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np

#from keras.models import sequential
from tensorflow.keras import Sequential
from keras.layers import Dense,Activation, Dropout
from keras.optimizers import SGD

# import tenserflow as tf
# from tenserflow.keras.models import Sequential
# from tenserflow.keras.layers import Dense
# from tenserflow.keras.layers import Dropout



import random

#initialize
words =[]
classes=[]
documents=[]
ignore_words=['?','!','@','$']

#use json
data_file = open('intents.json',encoding="utf-8").read()
intents = json.loads(data_file)

#populating the lists
for intent in intents['intents']: #it loops  1 intent (dictionary) amoung intents(list)
    for pattern in intent['patterns']:#it shows all the questions(pattern)
        
        
        #take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w) #w stores all the words in list
        
        
        #adding documents
        documents.append((w,intent['tag']))
        
        #adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
words = [ lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words ]

words = sorted(list(set(words))) #sorted in alphabetic order

classes = sorted(list(set(classes)))

#print(len(documents),'Documents: ', documents) #910
#print("\n")

#documents makes the words an array(touples)
#[(['Hi', 'there'], 'greeting'),
# (['How', 'are', 'you'], 'greeting')]

# print(len(classes),'classes: ', classes) #255
# print("\n")

#classes identifies the classes('tag')
#['Support Vector Machine', 'Decision Trees', etc]


# print(len(words),'unque limitized words: ', words) #642
# print("\n")

#words are unique words 
#["'", "'action", "'ll", "'m", "'s", "'value",
# '(', ')', ',', '.', ':', 'a', 'a/b', 'abnormal', 
# 'about', 'according', 'accuracy', 'act', 'action',
# 'action-value', 'adatset', 'advanategeous', 'advanatges',etc]

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

#initializing training data
training = []
output_empty = [0]*len(classes) #containts[0]*len(classses) if 2 cl then [0,0] 

for doc in documents:
    
    #initializing the bag of words
    bag = []
    
    #list of tokenized words for questions(pattern)
    pattern_words = doc[0] #doc[0]=(w,tag)==(['Hi', 'there'], 'greeting')
    
    #lemmatize eache words- creates base word,in agttempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    #for w in words(itterates for no of words{bag=[642 times]})
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
        #if w is in pattern_word(matches) then bag[1] else bag[0]
    
    #output is a '0' for all other tags and '1' for current tag(for each pattern)
    output_row = list(output_empty)#output_row=[list(leg(classes))]
    output_row[classes.index(doc[1])] = 1 #[tag.index_no[1]]:1 indes e dea delo tag er
    
    training.append([bag,output_row])
    #traning list will contain row pair of each bag and output row

#print("training: ",training)
    
random.shuffle(training)
training = np.array(training, dtype="object") #tenserflow only works swith numpy

train_x = list(training[:,0]) #extracts first column from array
train_y = list(training[:,1]) #extracts second column from array

# print("trainning data created")
# print("train_x: ",train_x)
# print("train_y: ",train_y)

model = Sequential()

#input layer
model.add(Dense(256, input_shape=(len(train_x[0]),)))#input_shape=(length of first train_x)
model.add(Activation('relu'))

#hidden layer 1
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5)) #trainning er somoe over fitting bachanor jono dropout

#hidden layer 1
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5)) #trainning er somoe over fitting bachanor jono dropout

#output layer
model.add(Dense(len(train_y[0])))#input_shape=(length of first train_y)
model.add(Activation('softmax'))

# #input layer
# #model.add(Dense(128, input_shape=(len(train_x[0])), activation ='relu'))
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))

# #input_shape=(length of first train_x)
# model.add(Dropout(0.5)) #trainning er somoe over fitting bachanor jono dropout
# model.add(Dense(64, activation ='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation ='softmax'))
# #input_shape=(length of first train_y)

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = tf.keras.optimisers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x),np.array(train_y), epochs=200, batch_size=5, verbose=1)

model.save('chatbot_model.h5',hist)

print('model created')
