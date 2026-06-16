# Student Performance Prediction System

## Project Overview

This project is a Machine Learning application that predicts a student's final exam score based on factors such as study hours, attendance percentage, previous academic performance, age, and sleep hours.

The main goal of this project is to understand how Machine Learning can be used to analyze educational data and predict student performance. A Random Forest Regression model was used to train and test the data.

This project was developed as part of my AI & ML Internship Minor Project.

---

## Problem Statement

Student performance depends on many factors, including study habits and attendance. It can be difficult to estimate how a student might perform in the final exam.

This project helps predict a student's expected final exam score using Machine Learning techniques.

---

## Features

* Load and analyze student data
* Perform data preprocessing
* Train a Machine Learning model
* Predict final exam scores
* Evaluate model accuracy
* Save the trained model
* Interactive Streamlit web application

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Streamlit
* Pickle

---

## Dataset Information

The dataset contains the following information:

* Student ID
* Gender
* Age
* Study Hours Per Week
* Attendance Percentage
* Previous Score
* Parental Education Level
* Internet Access
* Extracurricular Activities
* Sleep Hours
* Final Exam Score
* Pass/Fail Status

---

## Machine Learning Algorithm Used

**Random Forest Regressor**

Random Forest was chosen because it provides good accuracy and handles data effectively without requiring complex preprocessing.

---

## Steps Performed

1. Imported the dataset using Pandas.
2. Explored and understood the dataset.
3. Selected important features for prediction.
4. Split the dataset into training and testing sets.
5. Trained a Random Forest Regression model.
6. Predicted student exam scores.
7. Evaluated model performance using MAE and R² Score.
8. Saved the trained model using Pickle.
9. Created a Streamlit web application for user interaction.

---

## Model Performance

### Results

* Mean Absolute Error (MAE): **2.19**
* R² Score: **0.9557**

### Interpretation

The model achieved an R² score of 95.57%, which indicates that it predicts student performance with high accuracy.

---

## Project Structure

```text
student-performance-prediction
│
├── data
│   └── student_exam_performance.csv
│
├── src
│   └── train_model.py
│
├── images
│
├── model.pkl
├── app.py
|──correlation_matrix.py
├── requirements.txt
└── README.md
```

## How to Run the Project

### Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn streamlit
```

### Train the Model

```bash
python src/train_model.py
```

### Run the Streamlit Application

```bash
streamlit run app.py
```

---

## Sample Input

* Age: 17
* Study Hours Per Week: 15
* Attendance Percentage: 85
* Previous Score: 78
* Sleep Hours: 7

The model predicts the expected final exam score based on these inputs.

---

## Future Improvements

* Add more student-related features.
* Compare multiple Machine Learning algorithms.
* Deploy the application online.
* Add more visualizations and dashboards.
* Include pass/fail classification.

---

## Conclusion

This project demonstrates the use of Machine Learning for educational data analysis. The model successfully predicts student exam scores with high accuracy and provides a simple interface for users through Streamlit.

---

## Author

**Rashi Jaiswal**

AI & ML Internship Minor Project
