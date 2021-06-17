## Activation test .ipynb file

%load_ext tensorboard
%tensorboard --logdir logs

!pip uninstall tensorflow
!pip install tensorflow

## The AI
# the variables that will be tested
activations = ["linear", "elu", "relu", "selu", "tanh", "softmax", "softplus", "softsign", "sigmoid", "hard_sigmoid"]
learning_rates = [0.001, 0.0001, 0.01]
epochs = [100, 500]


import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('AIOptiver-8fe80fafc591.json', scope)
# connect to spreadsheet
client = gspread.authorize(creds)

main_sheet = client.open('activation').worksheet("main") # open the sheet where the data will be stored


for z in epochs:
  for y in learning_rates:
    for x in activations: # loop over every possibility
      index = 0
      if (y == 0.0001): #place to store the variables in spreadsheet
        index += 11
      elif (y == 0.01):
        index += 22
      if (z == 500):
        index += 33
      
      if (main_sheet.cell(index + activations.index(x)+2, 5).value == ''):
        print(x)
        # train the AI with variables and retrieve output
        a, b, c, d, history, model = TrainAI(x, y, z, "mean_squared_error")
        print(f"FirstLoss: {a}, LastLoss: {b}, FirstMAE: {c}, LastMAE: {d}, Activation: {x}, Learning_rate; {y}, epochs: {z}")


        # store output in spreadsheet
        try: # to catch errors so that the training will continue
          main_sheet.update_cell(index + activations.index(x)+2, 5, a)
          main_sheet.update_cell(index + activations.index(x)+2, 6, b)
          main_sheet.update_cell(index + activations.index(x)+2, 7, float(c))
          main_sheet.update_cell(index + activations.index(x)+2, 8, float(d))
        except:
          ...


        #####################################################################
        # create graph which shows training proces
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(3,6))

        fig, ax1 = plt.subplots()
        plt.title(f'Training using: {x}, learning_rate={y}')
        # do things
        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('MAE', color=color)
        ax1.plot(history.history['mean_absolute_error'], color=color, label="MAE")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0, 10000])
        ax2 = ax1.twinx()

        color = 'tab:blue'
        ax2.set_ylabel('Loss', color=color
        ax2.plot(history.history['loss'], color=color, label="Loss")
        ax2.tick_params(axis='y', labelcolor=color)

        fig.legend(loc='upper right', bbox_to_anchor=(0.865, 0.9), title="Legend")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        plt.savefig(f"graphs/training/{z}/plot_{x}_learning{str(y).split('.')[-1]}", format="png", dpi=fig.dpi) # save image
        plt.show()


        #####################################################################
        # predict look_ahead points (based on dataset)

        import numpy as np
        testdata = np.load("npy/testdata_100_10.npy") # open test file
        print(testdata.shape)

        # open scaler
        from sklearn.externals import joblib
        scaler_filename = "scaler.save"
        scaler = joblib.load(scaler_filename) 

        testdata[0,0] = scaler.transform(np.array([testdata[0,0]]))
        print(testdata.shape)

        look_ahead = 10
        amount_of_points = 100
        batch_size = 1
        trainPredict = np.array([testdata[0][0:,0:amount_of_points]])
        predictions = np.zeros((look_ahead))
        print(trainPredict.shape)
        for i in range(look_ahead): # predict 10 points
            prediction = model.predict(np.array([trainPredict[0][0:,0:]]), batch_size=batch_size)
            print(prediction)
            predictions[i] = prediction[0] # store prediction
            trainPredict = np.array([testdata[0][0:,i:amount_of_points+i]]) # set input data

        # make a graph of the predictions
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6)) # initiate graph
        plt.plot(np.arange(look_ahead),predictions,'r', label="prediction traffic")
        plt.plot(np.arange(look_ahead),np.load("npy/testdata_100_10.npy")[0][0, -look_ahead-1:-1], label="validation traffic")
        plt.xlabel("time_stamps")
        plt.ylabel("network_load (Mbps)")
        plt.title(f'Prediction using: {x}, learning_rate={y}')
        plt.legend()
        plt.savefig(f"graphs/predictions/{z}/plot_{x}_learning{str(y).split('.')[-1]}_prediction", format="png", dpi=fig.dpi) # save image
        plt.show()
        
## Part 2        
        
 def TrainAI(activation, learning, epochs, loss):
  # 
  import tensorflow as tf

  from google.colab import output
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.layers import Reshape
  from tensorflow.keras.layers import LSTM
  import numpy as np


  trainX = np.load("npy/newdata.npy")
  trainY = np.load("npy/newdataoutput.npy")

  trainY = np.reshape(trainY[0], (64,1))
  batch_size = 1
  amount_of_points = 100


  from sklearn.externals import joblib
  scaler_filename = "scaler.save"
  scaler = joblib.load(scaler_filename) 

  trainX[0:,0] = scaler.transform(trainX[0:,0])


  model = Sequential()

  model.add(LSTM(64, input_shape=(2, amount_of_points), return_sequences=True, activation=activation)) # add first layer
  model.add(LSTM(32, activation=activation)) # add second layer

  model.add(Dense(1, activation=activation)) # add last layer

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning) # define optimizer
  model.compile(loss=loss, optimizer=optimizer, metrics=["mae"]) # compile model


  model.summary()

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)

  import time
  start_time = time.time()
  history = model.fit(np.array(trainX), np.array(trainY), epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, callbacks=[tensorboard_callback]) # train AI
  # output.clear()
  print("--- fitting model took {} seconds / {} minutes ---".format(int(time.time()-start_time), round((time.time()-start_time)/60, 2))) # print the time it took to execute
  return (history.history['loss'][0], history.history['loss'][-1], history.history['mean_absolute_error'][0], history.history['mean_absolute_error'][-1], history, model)

