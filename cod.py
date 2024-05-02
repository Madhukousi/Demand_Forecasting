import pandas as pd

df=pd.read_csv('Daily_Demand_Forecasting_Orders.csv')

print(df.head())

print(df.columns)

print(df.isnull().sum())

y=df['Target (Total orders)']

x=df.drop(columns=['Target (Total orders)','Unnamed: 0'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

from sklearn.tree import DecisionTreeRegressor
d=DecisionTreeRegressor(criterion="squared_error")

print(d.fit(x_train,y_train))

print(d.score(x_test, y_test))

pred_y=d.predict(x_test)

print((pred_y-y_test).abs().mean()*1000)  #abs is where it always return positive number of that number

regression_tree=DecisionTreeRegressor(criterion="absolute_error")
print(regression_tree.fit(x_train,y_train))

print("score:",regression_tree.score(x_test,y_test))
pred=regression_tree.predict(x_test)
print("mae:",((pred_y-y_test).abs().mean()*1000))