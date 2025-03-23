
#Init

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score



df = pd.read_csv('gameandgrade.csv')
df.head()


#Prapare Data 
df['Grade'] = pd.to_numeric(df['Grade'], errors='coerce')
print(df.dtypes)

def grade_to_scale(grade):
  if 0 <= grade <= 30:
    return 1
  elif 30 < grade <= 50:
    return 2
  elif 50 < grade <= 65:
    return 3
  elif 65 < grade <= 80:
    return 4
  elif grade > 80:
    return 5
  else:
    return None  

df['Scale'] = df['Grade'].apply(grade_to_scale)


print(df[['Grade', 'Scale']])


#df['Grade'] = pd.to_numeric(df['Grade'], errors='coerce')
#print(df.dtypes)

#def grade_to_scale(grade):
 # return 0.04 * grade + 0.71  


#df['Scale'] = df['Grade'].apply(grade_to_scale)


#print(df[['Grade', 'Scale']])

#Frist model 
features = ["Sex", "School Code", "Playing Years", "Playing Often", "Playing Hours", "Playing Games","Father Education" , ]

X = df[features]
y = df['Scale']
df.dropna(subset=['Scale'] + features, inplace=True)
X = df[features]
y = df['Scale']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# Create a linear regression model
model = RandomForestRegressor(max_depth=15, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.show()



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# You can also calculate other metrics like R-squared, etc.
# For example:
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)


#second model
features = ["Sex", "School Code", "Playing Years", "Playing Often", "Playing Hours", "Playing Games","Father Education",'Scale' ]
X = df[features]
y = df['Grade']
df.dropna(subset=['Grade'] + features, inplace=True)
X = df[features]
y = df['Grade']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use GradientBoostingRegressor for continuous target variables
model = GradientBoostingRegressor(max_depth=5, random_state=0)

# Train the model
model = GradientBoostingRegressor(n_estimators=90, learning_rate=0.1, max_depth=2, random_state=42) # Changed to GradientBoostingRegressor
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.show()



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# You can also calculate other metrics like R-squared, etc.
# For example:
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)



#Third model
features = ["Sex", "School Code", "Playing Years", "Playing Often", "Playing Hours", "Playing Games","Father Education",	'Mother Education' ]
X = df[features]
y = df['Scale']
df.dropna(subset=['Scale'] + features, inplace=True)
X = df[features]
y = df['Scale']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use GradientBoostingRegressor for continuous target variables
model = GradientBoostingRegressor(max_depth=5, random_state=0)

# Train the model
model = GradientBoostingRegressor(n_estimators=90, learning_rate=0.1, max_depth=2, random_state=42) # Changed to GradientBoostingRegressor
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Actual vs Predicted Grades")
plt.show()



# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# You can also calculate other metrics like R-squared, etc.
# For example:
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)