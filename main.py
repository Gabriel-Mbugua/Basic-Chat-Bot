import nltk
from nltk.stem.lancaster import LancasterStemmer
# nltk.download('punkt')

stemmer = LancasterStemmer()

import numpy as np
import tflearn.estimators.cluster
import tensorflow as tf
import random
import json
import pickle as pk

with open("intents.json") as file:
        data = json.load(file)

#Purpose of try-except is to minimise number of times code is run if exsiting saves are present.
try: 
    #reads from an existing save with the data below. Note: "rb" means read bytes.
    with open("data.pickle", "r") as f:
        words, labels, training, output = pk.load(f)
    print("*****Opening existing*****")
    
except:
    # print(data["intents"])
    print("*****Using loaded data*****")

    words = []
    labels = []
    docs_x = []#pattern
    docs_y = []#tag that pattern above belongs in

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
                labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words))) #removes duplicate words and sorts them. "set" performs this.

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)    

        output_row = out_empty[:] 
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag) 
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    #writes a new save with the data below. Note: "wb" means write bytes.
    with open("data.pickle", "wb") as f:
        pk.dump((words,labels, training,output), f)

tf.reset_default_graph()#simply resets previous graph

#Neural network
#start with input data that's set to the same length as the training data
net = tflearn.input_data(shape=[None, len(training[0])])
#Hidden layers
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#Output layer. "Softmax" allows us to get probabilities for each output
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#Don't train model if we already have a model that exists
try:
    # model.load("model.tflearn")
    # Uncomment this to retrain your model and comment out the line above
    me.py 
except:
    #Feeding our model data
    #n_epoch meanss the number of times we'll have our model look at the data
    model.fit(training, output, n_epoch=1000,batch_size=8, show_metric=True)
    model.save("model.tflearn")

#convert user input into a bag of words
def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for s_wrd in s_words:
        for i, w in enumerate(words):
            if w == s_wrd:
                bag[i] = 1

    return np.array(bag)

#ask user for input then return a response  
def chat():
    print("Start talking with the bot! (Type quit to stop)")
    while True:
        inp =input("You :")
        if inp.lower() == "quit":
            break
        
        results = model.predict([bagOfWords(inp, words)])[0]
        results_index = np.argmax(results) #retrieve index of the most probable result
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                # if tg['tag'] == "goodbye":
                #     print("Bye")
                #     return
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
 
        else:
            print("Sorry, I'm not sure what you mean. Try again!")
        
        

chat()