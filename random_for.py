import numpy as np
import pickle

model_load=pickle.load(open(r'C:\Users\KIIT0001\Documents\GitHub\Mini_project_ML\random_forest_model.sav','rb'))

input_data=(45,30.5,140,1,1,0)
# changing the input data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

# reshape the array as we are predicting for one instance   
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=model_load.predict(input_data_reshaped)
print(prediction)
if (prediction[0]==0):
  print('The person is not at risk of diabetes')
else:  print('The person is at risk of diabetes')
