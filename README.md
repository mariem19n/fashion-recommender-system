# 👗 Fashion Recommender System (Hybrid Recommendation Engine)

This project implements a **hybrid recommendation system** for fashion products  
(similar to SHEIN, Zara, H&M-style product discovery).  
The recommender combines **Content-Based**, **Popularity-Based**,  
**Constraint-Based**, and **Knowledge-Based** techniques inside a modern Streamlit app.

---

## 🚀 Features

### 🔍 Recommendation Models
- **Content-Based Filtering**  
  Recommends similar items based on product features (category, price, color, brand).
- **Popularity-Based Filtering**  
  Ranks products by rating, number of views or purchases.
- **Constraint-Based Filtering**  
  User selects filters such as category, color, price range → shown like an e-commerce interface.
- **Knowledge-Based Filtering**  
  Rule-based suggestions (e.g., “Winter items”, “Budget items”, “Premium picks”).
- **Reference Product Choice (optional)**  
  User can select a reference item → recommendations are computed from it.  
  Includes a “None” option to disable this part.

---

## 🎨 User Interface (SHEIN-style)

The app provides a clean and simple e-commerce-like UI:

- Centered filters and criteria  
- Modern cards for results  
- Clear product display (name, brand, price, rating)  
- Responsive layout

---

## 📂 Project Structure

```

SYST_REC/
│── app.py                    # Streamlit application
│── src/
│     ├── exploration.ipynb   # Data exploration & preprocessing
│     └── recommender_systems.ipynb
│── requirements.txt
│── README.md
└── .gitignore

```

> ⚠️ The `data/` folder is not included in the repository.  
> Place your dataset here:
> `data/fashion_products.csv`

---

## 📊 Dataset

The system uses a CSV dataset of fashion products containing:

- Product ID  
- Product Name  
- Brand  
- Category  
- Price  
- Color  
- Size  
- Rating  

Add your dataset as:

```

data/fashion_products.csv

````

---

## ▶️ Run the Application

### 1. Install dependencies

```bash
pip install -r requirements.txt
````

### 2. Launch the Streamlit app

```bash
streamlit run app.py
```

The app opens in your browser.

---

## 🧠 How Recommendations Work

### Content-Based

Uses similarity between product features to find related items.

### Popularity-Based

Ranks items by rating or demand.

### Constraint-Based

Filters chosen by the user (like in fashion websites).

### Knowledge-Based

Simple expert rules (price constraints, seasonal items, etc.).

The system does **not** use deep learning → ideal for smaller datasets.

---

