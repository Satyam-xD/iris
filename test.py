import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris=load_iris()
x = iris.data
y = iris.target
labels=iris.target_names
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.40,random_state=0)
model = pickle.load(open('model.pkl','rb'))


model=LogisticRegression()

model.fit(X_train,y_train)
pickle.dump(model,open('model.pkl','wb'))

y_pred=model.predict(X_test)
accuracy= accuracy_score(y_test,y_pred)
print("score",accuracy)
