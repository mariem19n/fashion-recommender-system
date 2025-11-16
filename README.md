# 👗 Fashion Hybrid Recommendation System (Content + Constraint + Knowledge Based)

This project implements a **hybrid recommendation system** for fashion products  
(similar to SHEIN, Zara, H&M product discovery).

The model combines:

- **Content-Based Filtering** (vector similarity using One-Hot + Price + Rating)
- **Constraint-Based Filtering** (user-selected filters like price, color, size…)
- **Knowledge-Based Rules** (rules for “sport”, “casual”, “premium”, “cheap”, etc.)
- **Hybrid Model** combining the three with weights (default: 0.6 / 0.2 / 0.2)

This repository also includes:

- A complete **evaluation pipeline** (Precision@K, Recall@K)
- A **heatmap** of hybrid similarities
- A **Streamlit app (app.py)** with a SHEIN-like UI

---

## 🚀 Main Features

### 🔵 1. Content-Based Filtering
- Builds a **product feature matrix** using:
  - Brand  
  - Category  
  - Color  
  - Size  
  - Price  
  - Rating  
- One-Hot Encoding for categorical features  
- Min-Max scaling for numerical features  
- Similarity = **cosine similarity**

### 🔶 2. Constraint-Based Filtering
User-selected constraints:
- Max price  
- Min price  
- Brand  
- Category  
- Color  
- Size  
- Min rating  
- Or automatically derived from a **reference product**  

Recommendations are sorted by:
- Highest rating
- Lowest price

### 🟢 3. Knowledge-Based Rules
Rules based on:
- **Usage** (sport, chic, casual, work)
- **Budget level** (cheap, mid, premium)
- Brand style
- Color preferences
- Size preferences
- Quality threshold (min rating)

Example rules:
- “Sport” → only Adidas or Nike  
- “Chic” → Gucci, black/red/gold colors  
- “Work” → neutral colors + selected brands  
- Budget segmentation using price quantiles

### 🔴 4. Hybrid Model
Final score combines:

```

hybrid_score =
0.6 * content_score +
0.2 * constraint_score +
0.2 * knowledge_score

```

Constraint & Knowledge scores ∈ {0,1}  
Content-Based score normalized ∈ [0,1]

---

## 📊 Evaluation Metrics Included

You have full evaluation for:

- **Content-Based Precision@K & Recall@K**
- **Hybrid Precision@K & Recall@K**
- **Global evaluation on a sample of products**
- **Visual comparison bar chart**
- **Heatmap of hybrid similarities**

---

## 📂 Project Structure

```

SYST_REC/
│── app.py                    # Streamlit UI 
│── src/
│     ├── exploration.ipynb   # Exploratory analysis
│     └── recommender_systems.ipynb  # All models + evaluation
│── data/                     # (IGNORED IN GIT)
│     └── fashion_products.csv
│── requirements.txt
│── .gitignore
└── README.md

```



---

## ▶️ Running the Streamlit App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📘 Technologies Used

* Python
* Pandas / NumPy
* Scikit-Learn
* Streamlit
* Seaborn / Matplotlib

---


