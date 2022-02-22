import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("Iris.csv")

print(df.head())

# Select independent and dependent variable
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Feature scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test= sc.transform(X_test)

# Instantiate the model
# model = RandomForestClassifier()
model = KNeighborsClassifier()
# Fit the model
model.fit(X_train, y_train)

# Make pickle file of our model
# dump(classifier, open("model.pkl", "wb"))

filename = 'savemodel.pkl'
pickle.dump(model, open(filename, 'wb'))

load_model = pickle.load(open(filename, 'rb'))

print(load_model.predict([[6.0, 2.2, 4.0, 1.0]]))