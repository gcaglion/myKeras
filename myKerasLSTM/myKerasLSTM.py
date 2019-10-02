# multivariate output stacked lstm example
import numpy
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import cx_Oracle

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return numpy.array(X), numpy.array(y)

trainDate0="2018-05-30-00:00"
trainHistoryLen=15000
inferDate0="2018-06-30-00:00"
inferHistoryLen=80

con=cx_Oracle.connect('History/HistoryPwd@Algo')

cur=con.cursor()
hstmt="select open,high,low,close from( select newdatetime, open, lead(open) over(order by newdatetime) openF, high, lead(high) over(order by newdatetime) highF, low, lead(low) over(order by newdatetime) lowF, close, lead(close) over(order by newdatetime) closeF from ( select newdatetime, open, high, low, close from eurusd_h1 where newdatetime<=to_date('"+trainDate0+"','YYYY-MM-DD-HH24:MI') order by 1 desc) where rownum<"+str(trainHistoryLen)+" order by newdatetime)"
cur.execute(hstmt)
resList=cur.fetchall()
trainSet=numpy.asarray(resList)
cur.close()

cur=con.cursor()
hstmt="select open,high,low,close from( select newdatetime, open, lead(open) over(order by newdatetime) openF, high, lead(high) over(order by newdatetime) highF, low, lead(low) over(order by newdatetime) lowF, close, lead(close) over(order by newdatetime) closeF from ( select newdatetime, open, high, low, close from eurusd_h1 where newdatetime<=to_date('"+inferDate0+"','YYYY-MM-DD-HH24:MI') order by 1 desc) where rownum<"+str(inferHistoryLen)+" order by newdatetime)"
cur.execute(hstmt)
resList=cur.fetchall()
inferSet=numpy.asarray(resList)
cur.close()

con.close()
 
#print(trainSet)

# choose a number of time steps
n_steps = 50
# split training set into samples
X, y = split_sequences(trainSet, n_steps)
#print(X.shape, y.shape)

# summarize the data
#for i in range(len(X)):
#	print(X[i], y[i])

# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')

# fit model
model.fit(X, y, epochs=5, verbose=1)

# split infer set into samples
X, y = split_sequences(inferSet, n_steps)
print(y)
print("-------------------------------------------------------------------")
# demonstrate prediction
yhat = model.predict(X, verbose=1)
print(yhat)
