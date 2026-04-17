# netflix-recommendation-system
Netflix-style recommender using NeuMF
# 🎬 Netflix Recommendation System (NeuMF)

A Netflix-style movie recommendation system built using **Neural Collaborative Filtering (NeuMF)**, combining deep learning with user-item interaction modeling.

---

## 🚀 Features

* Personalized movie recommendations
* Neural Matrix Factorization (GMF + MLP)
* Trained on MovieLens 1M dataset
* Filters already watched movies
* Interactive web app using Streamlit
* Adjustable Top-K recommendations

---

## 🧠 Model Architecture

This project uses **NeuMF (Neural Matrix Factorization)**:

* **GMF (Generalized Matrix Factorization)** → captures linear interactions
* **MLP (Multi-Layer Perceptron)** → captures non-linear patterns
* **Fusion Layer** → combines both for final prediction

---

## 🖥️ Tech Stack

* Python
* TensorFlow / Keras
* Pandas / NumPy
* Streamlit

---

## 📥 Dataset

This project uses the **MovieLens 1M dataset**.

👉 Download from:
https://grouplens.org/datasets/movielens/

After downloading, place it like:

```
ml-1m/
   movies.dat
   ratings.dat
```

⚠️ Note: Full dataset is not included due to size constraints.

---

## ▶️ How to Run

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run the app

```
streamlit run app.py
```

---

## 📊 Output

* Enter a user ID
* Get top movie recommendations
* Recommendations exclude already watched movies

---

## ⚠️ Notes

* `model.h5` is not uploaded due to size limits
* You can retrain the model using the provided code

---

## 👨‍💻 Author

**Lakshit Saraswat**
🔗 LinkedIn: https://linkedin.com/in/LakshitSaraswat

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
