from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

#Matplotlib thingies
plt.rc('figure', autolayout = True)
plt.rc('axes', labelweight = 'bold', labelsize = 'large', titlesize = 18, titlepad = 10)

#Gets the airindex.csv file, removes unnecessary columns, drops rows which have a None value
df = pd.read_csv("air_index.csv")
df = df[["2022", "2021", "2020", "2019", "2018", "2017"]]
df = df.dropna().reset_index(drop=True)


#Takes 70% of the data as training data randomly
df_train = df.sample(frac=0.7, random_state=0)

#Drops the training data from the databse, df_valid now only contains the rows which aren't present in the training data
df_valid = df.drop(df_train.index)


#X_train gets the columns (except predicted column basically) which the data will train on from the training dataset, 
# X_valid gets the same columns but from the valid dataset
X_train = df_train.drop('2022', axis=1)
X_valid = df_valid.drop('2022', axis = 1)


#y_train will get the column we want to predict i.e 2022's airquality from the training dataset
#y_valid will get the column we want to predict i.e 2022's airquality from the validation dataset
y_train = df_train['2022']
y_valid = df_valid['2022']


#Defining the architecture of the neural model, in this case we have 1 input layer, 3 hidden layer and 1 output layer
model = keras.Sequential([
    layers.Dense(32, activation = 'relu', input_shape = [5]),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(128, activation = 'relu'),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(1)
])


#Giving the loss and optimization functions
model.compile(optimizer='adam', loss='mae')


#Defining early stopping to stop overfitting
early_stopping = EarlyStopping(min_delta=0.01, patience=5, restore_best_weights=True)


#Fitting the model according to the practice data and the giving it the validation data 
#Also doing 100 epochs and with a batch size of 300
history =  model.fit(
    X_train, y_train,
    validation_data = (X_valid, y_valid),
    batch_size = 300,
    epochs = 100,
    callbacks =[early_stopping]
)

#Converts the history.history into a panda dataframe so we can view and track the loss and validation loss as more and more 
#epochs go on
history_df = pd.DataFrame(history.history)
history_df['loss'].plot()
history_df['val_loss'].plot()
plt.show()

#Saves the model
model.save("airquality2022model.model")
