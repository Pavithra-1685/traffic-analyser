import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib 

#analysing and reading data


df = pd.read_csv('Traffic Accident analyser/traffic_data_large.csv')
def data_analysis():
  print(df.head())
  print(df.isnull().sum())
  df.dropna(inplace=True)

def encoding():
  if df['weather'].dtype == 'object':
    df['weather'] = df['weather'].astype('category').cat.codes
    df['road_type'] = df['road_type'].astype('category').cat.codes
    df['hour'] = df['hour'].astype('category').cat.codes
    df['accident_occurred'] = df['accident_occurred'].astype('category').cat.codes


X = df[['hour', 'weather', 'road_type', 'vehicle_count']]
y = df['accident_occurred']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy_Score: ", accuracy)
print(f"Model is Trained and it's accuracy scored at {accuracy} percentage" )




# Create a 1x3 subplot (1 row, 3 columns)
fig, axs = plt.subplots(1, 3, figsize=(10, 5))

# ---- 1. Correlation Heatmap ----
def corr():
  sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=axs[0])
  axs[0].set_title("Feature Correlation Heatmap")

# ---- 2. Countplot: Predicted vs Actual ----
def countplot():
  sns.countplot(x=y_pred, hue=y_test, ax=axs[1])
  axs[1].set_title("Predicted vs Actual Accidents")
  axs[1].set_xlabel("Predicted")
  axs[1].legend(title='Actual')

# ---- 3. Confusion Matrix ----
def cm():
  cm = confusion_matrix(y_test, y_pred)
  sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axs[2])
  axs[2].set_title("Confusion Matrix")
  axs[2].set_xlabel("Predicted")
  axs[2].set_ylabel("Actual")

# Adjust layout and show all
plt.tight_layout()
plt.show()

# ---- 4. Classification Report ----
def cr():
  print("Classification Report:")
  print(classification_report(y_test, y_pred))

#---save the model----
joblib.dump(model, 'logistic_model.pkl')