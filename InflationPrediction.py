import os
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename

import math
import numpy as np
import pandas as pd
import sklearn.metrics
import statsmodels.api as sm
import xgboost as xgb
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# method called after uploading input csv
def process_model():
    global v
    global csv_file_path
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    xg_plot_button['state'] = tk.NORMAL
    keras_plot_button['state'] = tk.NORMAL


def xg_plot():
    df = pd.read_csv(csv_file_path)
    all_cols = list(df.columns)
    inflation_col = all_cols[-1]  # Inflation is last column
    all_cols.remove(inflation_col)
    feature_cols = all_cols[2:36]  # ignore month and year columns

    x = df[feature_cols]
    y = df[inflation_col]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)

    xgbr = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.01,
                            max_depth=4, alpha=0, n_estimators=1200, early_stopping_rounds=10, n_jobs=6, verbosity=1)

    model = xgbr.fit(xtrain, ytrain, eval_metric=["error", "logloss"])
    score = xgbr.score(xtrain, ytrain)
    print("Training score: ", score)

    newWindow = Toplevel(window)
    # sets the title of the
    # Toplevel widget
    newWindow.title("XGBOOST")
    # sets the geometry of toplevel
    newWindow.geometry("1200x800")
    newWindow.configure(bg='sky blue')
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, "Training score: " + str(score))

    scores = cross_val_score(xgbr, xtrain, ytrain, cv=10)
    print("Mean cross-validation score: %.2f" % scores.mean())
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, "Mean cross-validation score: %.2f" % scores.mean())

    rng = np.random.RandomState(31337)
    kfold = KFold(n_splits=10, random_state=rng, shuffle=True)
    kf_cv_scores = cross_val_score(xgbr, xtrain, ytrain, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, "K-fold CV average score: %.2f" % kf_cv_scores.mean())

    ypred = xgbr.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    print("MSE: %.2f" % mse)
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, "MSE:" + str(mse))
    print("R square (R^2):" + str(sklearn.metrics.r2_score(ytest, ypred)))
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, "R square (R^2):" + str(sklearn.metrics.r2_score(ytest, ypred)))

    x = range(len(ytest))
    # the figure that will contain the plot
    fig = Figure(figsize=(6, 3),
                 dpi=100)
    plt = fig.add_subplot(111)
    plt.set_title('Inflation test and predicted data')
    plt.set_xlabel('Predicted')
    plt.set_ylabel('Original')
    plt.plot(x, ytest, label="Original")
    plt.plot(x, ypred, label="Predicted")
    plt.legend()

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master=newWindow)
    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    # plotting Extreme Gradient Boosting Prediction vs Observed
    fig2 = Figure(figsize=(10, 5), dpi=100)
    plt = fig2.add_subplot(111)
    plt.set_title('Extreme Gradient Boosting: Prediction Vs Test Data')
    plt.set_xlabel('Observed Inflation Output')
    plt.set_ylabel('Predicted Inflation Output')
    lowess = sm.nonparametric.lowess
    test = pd.DataFrame({"prediction": ypred, "observed": ytest})
    z = lowess(ypred, ytest)

    plt.plot(z[:, 1], z[:, 0], color="blue", lw=3)
    plt.scatter(ypred, ytest, color="red")
    canvas = FigureCanvasTkAgg(fig2, master=newWindow)
    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

    toolbar = NavigationToolbar2Tk(canvas, newWindow)
    toolbar.update()
    # browse_button.place(x=1050, y=10)
    # keras_plot_button.place(x=1050, y=50)
    # close_button.place(x=1050, y=90)


# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)


# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def keras_plot():
    df = pd.read_csv(csv_file_path)
    all_cols = list(df.columns)
    inflation_col = all_cols[-1]  # Inflation is last column
    all_cols.remove(inflation_col)
    feature_cols = all_cols

    x = df[feature_cols]
    y = df[inflation_col]

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=123)
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    input_dim = len(xtrain[0])
    output_dim = 1

    model = Sequential()

    model.add(Dense(22 * 4, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.1))


    model.add(Dense(22, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(22, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error",
                  metrics=["mean_squared_error", rmse, r_square, 'accuracy'])

    result = model.fit(xtrain, ytrain, epochs=1200, batch_size=32, verbose=1, validation_split=0.01)
    y_pred = model.predict(xtest)

    newWindow = Toplevel(window)
    # sets the title of the
    # Toplevel widget
    newWindow.title("Keras")
    # sets the geometry of toplevel
    newWindow.geometry("1200x800")
    newWindow.configure(bg='sky blue')
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, "R square (R^2):" + str(sklearn.metrics.r2_score(ytest, y_pred)))

    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END, ("Mean squared error (MSE):       %f" % sklearn.metrics.mean_squared_error(ytest, y_pred)))
    T = tk.Text(newWindow, height=2, width=100)
    T.pack()
    T.insert(tk.END,
             "Root mean squared error (RMSE): %f" % math.sqrt(sklearn.metrics.mean_squared_error(ytest, y_pred)))

    # plot training curve for R^2 (beware of scale, starts very low negative)
    # the figure that will contain the plot
    fig1 = Figure(figsize=(6, 5),
                  dpi=100)
    plt = fig1.add_subplot(111)
    regressor = LinearRegression()
    regressor.fit(ytest.values.reshape(-1, 1), y_pred)
    y_fit = regressor.predict(y_pred)
    reg_intercept = round(regressor.intercept_[0], 4)
    reg_coef = round(regressor.coef_.flatten()[0], 4)
    reg_label = "y = " + str(reg_intercept) + "*x +" + str(reg_coef)
    plt.scatter(ytest, y_pred, color='blue', label='data')
    plt.plot(y_pred, y_fit, color='red', linewidth=2, label='Linear regression\n' + reg_label)
    plt.set_title('Linear Regression')
    plt.legend()
    plt.set_xlabel('Observed')
    plt.set_ylabel('Predicted')

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig1, master=newWindow)

    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    # plot Predicted vs Observed
    fig2 = Figure(figsize=(10, 5),
                  dpi=100)
    plt = fig2.add_subplot(111)
    plt.set_title('Keras: Prediction Vs Test Data')
    plt.set_xlabel('Observed Inflation Output')
    plt.set_ylabel('Predicted Inflation Output')
    plt.plot(ytest.values, color='red', label="Observed Values", marker='x')
    plt.plot(y_pred, color='green', label="Predicted Values", linestyle=':', marker='+')

    plt.legend()
    canvas = FigureCanvasTkAgg(fig2, master=newWindow)

    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, newWindow)
    toolbar.update()
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    # browse_button.place(x=1050, y=10)
    # xg_plot_button.place(x=1050, y=50)
    # close_button.place(x=1050, y=90)



# print(sys)
# The main tkinter window
window = Tk()
# setting the title and
window.title('Inflation Prediction')
# setting the dimensions of

# the main window
window.geometry("1100x100")
v = tk.StringVar()
entry = tk.Entry(window, textvariable=v)


browse_button = tk.Button(master=window, bg='black', fg='green', command=process_model, width=10,
                              text="Browse Data")

close_button = Button(window, bg='black', fg='blue', text='Close', width=10, command=window.destroy)
# button that displays the plot
xg_plot_button = Button(master=window, command=xg_plot, bg='black', fg='green', text="Run XGBoost", width=10,
                            state=tk.DISABLED)

keras_plot_button = Button(master=window, bg='black', fg='green', command=keras_plot, text="Run Keras", width=10,
                               state=tk.DISABLED)

# place the button into the window
browse_button.place(x=100, y=10)
xg_plot_button.place(x=300, y=10)
keras_plot_button.place(x=500, y=10)
close_button.place(x=700, y=10)
# B1.place(x=1000, y=10)
window.configure(bg='sky blue')
window.mainloop()