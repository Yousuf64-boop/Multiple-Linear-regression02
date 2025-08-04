# Multiple Linear Regression from Scratch

This project demonstrates the implementation of **Multiple Linear Regression** using NumPy, without relying on libraries like scikit-learn for the model itself. The project includes:

- Custom Linear Regression class (`MeraLR`)
- Manual calculation of coefficients using the Normal Equation
- Model training and prediction
- R² Score evaluation using `sklearn.metrics`

---

## 📁 File Structure

- `Multiple_Linear_regression02.ipynb`: Jupyter Notebook containing full code, explanations, and testing of the custom model.

---

## 🧠 Key Concepts

- Matrix operations with NumPy
- Normal Equation for solving linear regression:
  
  \[
  \beta = (X^T X)^{-1} X^T y
  \]

- Manual handling of bias (intercept) using an extra column of ones
- Model evaluation using R² Score

---

## 🧪 How It Works

### `MeraLR` class

```python
class MeraLR:
    def fit(X_train, y_train)
    def predict(X_test)
Training
python
Copy code
lr = MeraLR()
lr.fit(X_train, y_train)
Prediction
python
Copy code
y_pred = lr.predict(X_test)
Evaluation
python
Copy code
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
✅ Example Output
yaml
Copy code
y_pred: [12. 14.]
R2 Score: 1.0
📦 Requirements
Python 3.x

NumPy

scikit-learn (only for metrics)

🚀 Future Work
Add support for gradient descent training

Extend to polynomial regression

Add mean squared error and visualizations

🧑‍💻 Author
Yousuf Midya
