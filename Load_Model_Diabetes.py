from numpy import loadtxt
from keras.models import load_model
import tkinter as tk
from tkinter import filedialog

# open saved model with filedialog and use it #

def main():
    # open file dialog for .csv data file:
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfile(title = "Select .csv data file", filetypes = (("CSV Files", "*.csv"), ))
    dataset = loadtxt(file_path.name, delimiter=',')
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    # open file dialog for model:
    file_path = filedialog.askopenfile()
    file_path = file_path.name
    # create model from h5 file:
    model = load_model(file_path)
    # summarize the model:
    model.summary()

    # evaluate loaded model on test data
    score = model.evaluate(X, Y, verbose=0)
    print("%s: %2.f%%" % (model.metrics_names[1], score[1]*100))

if __name__ == '__main__':
    main()