from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle 

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("----------------------------")
print("saving model as pickle file.")
pickle.dump(model, open("model.pkl", "wb"))
print("done!")