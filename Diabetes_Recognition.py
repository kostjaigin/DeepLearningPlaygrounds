from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import datetime as dt

# Train the model and save it to models folder #

def main():
    # values required to save model
    name = 'diabetes_recognition'
    date = dt.date.today()

    # load the dataset
    dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]

    # models in keras are defined as a sequence of layers
    # we add layers on at a time until we have our network architecture
    # define the keras model:
    model = Sequential()
    # the first hidden layer has 12 nodes and uses the relu activation function
    # in the same line we define the input layer with 8 inputs
    model.add(Dense(12, input_dim=8, activation='relu'))
    # the second hidden layer has 8 nodes and uses the relu
    model.add(Dense(8, activation='relu'))
    # the output layer has one node and uses the sigmoid function
    model.add(Dense(1, activation='sigmoid'))
    # we use cross entropy as the loss argument
    # adam algorithm as the optimizer
    # collect and report the classification accuracy defined with metrics arg
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # training occurs over epochs splitted into batches
    # epoch: one pass through all of the rows in the training dataset
    # batch: one or more samples considered by the model
    # within an epoch before weights are updated
    # these configurations can be chosen experimentally by trial-error

    # fit the keras model on the dataset
    # verbose set to 0 to hide outputs from each epoch
    model.fit(X, Y, epochs=400, batch_size=10)

    # model evaluation with training set
    _, accuracy = model.evaluate(X, Y)
    print('Accuracy: %.2f' % (accuracy*100))

    # plot the image of the network
    filename = "./models/" + name + "_" + date.strftime("%Y-%m-%d") + ".png"
    plot_model(model, to_file=filename)

    # # making probability predictions with the model
    # predictions = model.predict(X)
    # # we use sigmoid as output, which is why we will round the outcoming value
    # rounded = [round(x[0]) for x in predictions]

    # we can do the same as in commented code without rounding
    predictions = model.predict_classes(X)
    # summarize for the first 5 classes:
    for i in range(5):
        print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], Y[i]))

    # we can also save the trained model for future use:
    filename = "./models/" + name + "_" + date.strftime("%Y-%m-%d") + ".h5"
    model.save(filename)
    print("Model saved")



if __name__ == '__main__':
    main()