
# 🚦 Traffic Accident Analyzer (Logistic Regression)

This project analyzes and predicts traffic accident occurrences using a logistic regression model built with Python and scikit-learn. The application uses a traffic dataset with features like time, weather, road type, and vehicle count to predict whether an accident occurred or not.

---

## 📁 Folder Structure

```
Traffic Accident Analyzer/
├── traffic_data_large.csv         # Input dataset
├── app.py                         # Main Python script
├── logistic_model.pkl             # Saved Logistic Regression model
└── README.md                      # Project documentation (this file)
```

---

## 🧠 Features

* Data cleaning and encoding
* Logistic Regression model training
* Accuracy evaluation
* Correlation heatmap
* Prediction vs Actual countplot
* Confusion matrix visualization
* Classification report
* Model serialization using `joblib`

---

## 📦 Requirements

Install the necessary Python libraries using pip:

```bash
pip install pandas seaborn matplotlib scikit-learn joblib
```

---

## 🚀 How to Run

1. Place your dataset in a folder named `Traffic Accident analyser/` as `traffic_data_large.csv`.

2. Run the Python script:

```bash
python app.py
```

3. The script will:

   * Load and clean the data
   * Encode categorical features
   * Train a logistic regression model
   * Print accuracy and classification metrics
   * Visualize:

     * Correlation heatmap
     * Predicted vs Actual countplot
     * Confusion matrix
   * Save the trained model as `logistic_model.pkl`

---

## 📊 Dataset Features

| Feature             | Description                           |
| ------------------- | ------------------------------------- |
| `hour`              | Hour of the day                       |
| `weather`           | Weather condition (e.g. Rainy, Clear) |
| `road_type`         | Type of road (e.g. Highway, Urban)    |
| `vehicle_count`     | Number of vehicles at the scene       |
| `accident_occurred` | Binary label (1: Yes, 0: No)          |

---

## 📈 Model Performance

* **Accuracy**: Printed in the terminal after model training
* **Classification Report**: Includes precision, recall, F1-score
* **Visualization**:

  * Heatmap to show feature correlation
  * Countplot comparing predicted and actual values
  * Confusion matrix for error analysis

---

## 💾 Model Saving

After training, the logistic regression model is saved as:

```bash
logistic_model.pkl
```

You can load it in the future using:

```python
import joblib
model = joblib.load('logistic_model.pkl')
```

---

## 📌 Notes

* Ensure the dataset does not contain null values. The script automatically drops them.
* Only categorical fields like `weather`, `road_type`, and `hour` are encoded using label encoding.

---

## ✍️ Author

Pavithra H
ML & Web Development Enthusiast

[📬 Connect with Me](mailto:sculptureart457@gmail.com)


