import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
# 1. Chargement des données & préparation des features
# -------------------------------------------------------------------

@st.cache_data
def load_data(path: str = "data/fashion_products.csv") -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


@st.cache_data
def build_product_features(data: pd.DataFrame):
    """
    Crée :
      - products : 1 ligne par Product ID
      - feature_matrix : matrice de caractéristiques (pour Content-Based)
      - cheap_max, mid_max : seuils de prix (pour Knowledge-Based)
    """
    products = data.groupby("Product ID").agg({
        "Product Name": "first",
        "Brand": "first",
        "Category": "first",
        "Color": "first",
        "Size": "first",
        "Price": "mean",
        "Rating": "mean"
    }).reset_index()

    # Colonnes catégorielles et numériques
    cat_cols = ["Product Name", "Brand", "Category", "Color", "Size"]
    num_cols = ["Price", "Rating"]

    # Encodage One-Hot
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(products[cat_cols])

    # Normalisation des variables numériques
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(products[num_cols])

    # Matrice finale de features (pour content-based)
    feature_matrix = np.hstack([X_cat, X_num])

    # Seuils de prix pour Knowledge-Based
    price_q = products["Price"].quantile([0.33, 0.66])
    cheap_max = price_q.loc[0.33]
    mid_max = price_q.loc[0.66]

    return products, feature_matrix, cheap_max, mid_max


# -------------------------------------------------------------------
# 2. Fonctions de recommendation
# -------------------------------------------------------------------

# -------- CONTENT-BASED ------------------------------------------------------


def get_content_scores(product_id, products, feature_matrix) -> pd.Series:
    """Score de similarité cosinus pour tous les produits."""
    idx_list = products.index[products["Product ID"] == product_id].tolist()
    if not idx_list:
        raise ValueError(f"Product ID {product_id} not found.")
    idx = idx_list[0]

    query_vec = feature_matrix[idx].reshape(1, -1)
    sims = cosine_similarity(query_vec, feature_matrix)[0]

    return pd.Series(sims, index=products["Product ID"])


def recommend_content_based(product_id, products, feature_matrix, top_k=10) -> pd.DataFrame:
    sims = get_content_scores(product_id, products, feature_matrix)
    # ne pas recommander le même produit
    sims = sims.drop(labels=[product_id])
    top_ids = sims.sort_values(ascending=False).head(top_k).index

    recs = products[products["Product ID"].isin(top_ids)].copy()
    recs["similarity"] = recs["Product ID"].map(sims)
    recs = recs.sort_values("similarity", ascending=False)

    return recs


# -------- CONSTRAINT-BASED ---------------------------------------------------


def recommend_constraint_based(
    products,
    base_product_id=None,
    category=None,
    brand=None,
    max_price=None,
    min_price=None,
    color=None,
    size=None,
    min_rating=None,
    top_k=20
) -> pd.DataFrame:
    df = products.copy()

    # Si base_product_id est donné, on peut réutiliser sa Category/Brand par défaut
    if base_product_id is not None:
        base_row = products[products["Product ID"] == base_product_id]
        if not base_row.empty:
            if category is None:
                category = base_row["Category"].values[0]
            if brand is None:
                brand = base_row["Brand"].values[0]

    if category is not None:
        df = df[df["Category"] == category]
    if brand is not None:
        df = df[df["Brand"] == brand]
    if color is not None:
        df = df[df["Color"] == color]
    if size is not None:
        df = df[df["Size"] == size]
    if max_price is not None:
        df = df[df["Price"] <= max_price]
    if min_price is not None:
        df = df[df["Price"] >= min_price]
    if min_rating is not None:
        df = df[df["Rating"] >= min_rating]

    if base_product_id is not None:
        df = df[df["Product ID"] != base_product_id]

    df = df.sort_values(by=["Rating", "Price"], ascending=[False, True])

    return df.head(top_k)


# -------- KNOWLEDGE-BASED ----------------------------------------------------


def knowledge_based_recommend(
    products,
    cheap_max,
    mid_max,
    usage=None,          # "sport", "casual", "work", "chic"
    budget_level=None,   # "cheap", "mid", "premium"
    preferred_color=None,
    preferred_size=None,
    preferred_brand=None,
    min_rating=3.5,
    top_k=20
) -> pd.DataFrame:
    df = products.copy()

    # Règles de budget
    if budget_level == "cheap":
        df = df[df["Price"] <= cheap_max]
    elif budget_level == "mid":
        df = df[(df["Price"] > cheap_max) & (df["Price"] <= mid_max)]
    elif budget_level == "premium":
        df = df[df["Price"] > mid_max]

    # Règles d'usage
    if usage is not None:
        usage = usage.lower()

        if usage == "sport":
            sport_brands = ["Adidas", "Nike"]
            df = df[df["Brand"].isin(sport_brands)]

        elif usage == "casual":
            casual_brands = ["H&M", "Zara"]
            df = df[df["Brand"].isin(casual_brands)]

        elif usage == "work":
            work_colors = ["Black", "Grey", "Navy", "White"]
            df = df[df["Color"].isin(work_colors)]
            df = df[df["Brand"].isin(["H&M", "Zara", "Nike", "Adidas"])]

        elif usage == "chic":
            chic_colors = ["Black", "Red", "Gold", "Silver"]
            df = df[df["Color"].isin(chic_colors)]
            df = df[df["Brand"] == "Gucci"]
            df = df[df["Rating"] >= 4.0]

    # Préférences explicites
    if preferred_color is not None:
        df = df[df["Color"] == preferred_color]
    if preferred_size is not None:
        df = df[df["Size"] == preferred_size]
    if preferred_brand is not None:
        df = df[df["Brand"] == preferred_brand]

    df = df[df["Rating"] >= min_rating]
    df = df.sort_values(by=["Rating", "Price"], ascending=[False, True])

    return df.head(top_k)


# -------- HYBRID -------------------------------------------------------------


def get_constraint_scores_for_hybrid(
    products,
    base_product_id=None,
    category=None,
    brand=None,
    max_price=None,
    min_price=None,
    color=None,
    size=None,
    min_rating=None
) -> pd.Series:
    candidates = recommend_constraint_based(
        products=products,
        base_product_id=base_product_id,
        category=category,
        brand=brand,
        max_price=max_price,
        min_price=min_price,
        color=color,
        size=size,
        min_rating=min_rating,
        top_k=len(products)
    )
    valid_ids = set(candidates["Product ID"])
    scores = products["Product ID"].apply(lambda pid: 1.0 if pid in valid_ids else 0.0)
    scores.index = products["Product ID"]
    return scores


def get_knowledge_scores_for_hybrid(
    products,
    cheap_max,
    mid_max,
    usage=None,
    budget_level=None,
    preferred_color=None,
    preferred_size=None,
    preferred_brand=None,
    min_rating_kb=3.5
) -> pd.Series:
    candidates = knowledge_based_recommend(
        products=products,
        cheap_max=cheap_max,
        mid_max=mid_max,
        usage=usage,
        budget_level=budget_level,
        preferred_color=preferred_color,
        preferred_size=preferred_size,
        preferred_brand=preferred_brand,
        min_rating=min_rating_kb,
        top_k=len(products)
    )
    valid_ids = set(candidates["Product ID"])
    scores = products["Product ID"].apply(lambda pid: 1.0 if pid in valid_ids else 0.0)
    scores.index = products["Product ID"]
    return scores


def hybrid_recommend(
    base_product_id,
    products,
    feature_matrix,
    cheap_max,
    mid_max,
    # constraints
    category=None,
    brand=None,
    max_price=None,
    min_price=None,
    color=None,
    size=None,
    min_rating_constraint=None,
    # knowledge-based
    usage=None,
    budget_level=None,
    preferred_color_kb=None,
    preferred_size_kb=None,
    preferred_brand_kb=None,
    min_rating_kb=3.5,
    # poids FIXES
    w_content=0.6,
    w_constraint=0.2,
    w_knowledge=0.2,
    top_k=10
) -> pd.DataFrame:
    # Content-based
    content_scores = get_content_scores(base_product_id, products, feature_matrix)
    content_norm = content_scores - content_scores.min()
    denom = content_norm.max()
    if denom != 0:
        content_norm = content_norm / denom
    else:
        content_norm = content_norm * 0.0

    # Constraint scores
    constraint_scores = get_constraint_scores_for_hybrid(
        products=products,
        base_product_id=base_product_id,
        category=category,
        brand=brand,
        max_price=max_price,
        min_price=min_price,
        color=color,
        size=size,
        min_rating=min_rating_constraint
    )

    # Knowledge scores
    knowledge_scores = get_knowledge_scores_for_hybrid(
        products=products,
        cheap_max=cheap_max,
        mid_max=mid_max,
        usage=usage,
        budget_level=budget_level,
        preferred_color=preferred_color_kb,
        preferred_size=preferred_size_kb,
        preferred_brand=preferred_brand_kb,
        min_rating_kb=min_rating_kb
    )

    scores_df = pd.DataFrame({
        "Product ID": products["Product ID"],
        "content": content_norm.values,
        "constraint": constraint_scores.values,
        "knowledge": knowledge_scores.values
    }).set_index("Product ID")

    scores_df["hybrid_score"] = (
        w_content    * scores_df["content"] +
        w_constraint * scores_df["constraint"] +
        w_knowledge  * scores_df["knowledge"]
    )

    scores_df = scores_df[scores_df.index != base_product_id]

    result = products.merge(
        scores_df["hybrid_score"],
        left_on="Product ID",
        right_index=True
    )

    result = result.sort_values("hybrid_score", ascending=False)

    return result.head(top_k)


# -------------------------------------------------------------------
# 3. Helpers UI : cartes produits
# -------------------------------------------------------------------

def render_product_card(row):
    """Affiche une carte produit style e-commerce."""
    name = row["Product Name"]
    brand = row["Brand"]
    category = row["Category"]
    color = row["Color"]
    size = row["Size"]
    price = row["Price"]
    rating = row["Rating"]

    st.markdown(
        f"""
        <div class="product-card">
            <div class="product-image-placeholder">NEW</div>
            <div class="product-title">{name}</div>
            <div class="product-meta">{brand} • {category}</div>
            <div class="product-meta">Color: {color} • Size: {size}</div>
            <div class="product-bottom">
                <span class="product-price">{price:.2f} €</span>
                <span class="product-rating">⭐ {rating:.1f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------------
# 4. Interface Streamlit
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Fashion Recommender",
        page_icon="🛍️",
        layout="wide"
    )

    # CSS type Shein (header, cards, etc.)
    st.markdown(
        """
        <style>
        .main {
            background: #fafafa;
        }
        /* Header style */
        .top-bar {
            background: linear-gradient(90deg, #ff6f91, #ff9671, #ffc75f);
            padding: 14px 24px;
            border-radius: 0 0 16px 16px;
            color: #fff;
            margin-bottom: 20px;
        }
        .top-bar-title {
            font-size: 26px;
            font-weight: 800;
            letter-spacing: 1px;
        }
        .top-bar-sub {
            font-size: 13px;
            opacity: 0.9;
        }
        /* Product cards */
        .product-card {
            background-color: #ffffff;
            border-radius: 14px;
            padding: 10px 10px 12px 10px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.04);
            border: 1px solid #f0f0f0;
            margin-bottom: 14px;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .product-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        }
        .product-image-placeholder {
            background: #ffe1ec;
            color: #ff3366;
            border-radius: 10px;
            padding: 4px 8px;
            font-size: 11px;
            display: inline-block;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .product-title {
            font-weight: 650;
            font-size: 15px;
            margin-bottom: 4px;
        }
        .product-meta {
            font-size: 12px;
            color: #666;
            margin-bottom: 2px;
        }
        .product-bottom {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
        }
        .product-price {
            font-weight: 700;
            color: #ff3366;
            font-size: 14px;
        }
        .product-rating {
            font-size: 12px;
            color: #f39c12;
        }
        /* Filter box */
        .filter-box {
            background-color: #ffffff;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
            border: 1px solid #f0f0f0;
        }
        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .section-subtitle {
            font-size: 12px;
            color: #777;
            margin-bottom: 14px;
        }
        .small-tag {
            display: inline-block;
            background: #fff0f6;
            color: #ff2e63;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 11px;
            margin-left: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header façon Shein
    st.markdown(
        """
        <div class="top-bar">
            <div class="top-bar-title">FASHION RECOMMENDER</div>
            <div class="top-bar-sub">
                Inspirez-vous comme sur un site de shopping : filtres, profil utilisateur, et modèles de recommandation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-title">Vue générale</div>
        <div class="section-subtitle">
        Le <b>produit de référence</b> est utilisé comme point de départ pour les modèles
        <b>Content-Based</b> et <b>Hybrid</b>. Vous pouvez aussi choisir <i>(aucun)</i> pour tester
        seulement les modèles <b>Constraint-Based</b> et <b>Knowledge-Based</b>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Charger les données
    data = load_data()
    products, feature_matrix, cheap_max, mid_max = build_product_features(data)

    # Sidebar : modèle + top_k
    st.sidebar.header("⚙️ Paramètres")
    model_type = st.sidebar.selectbox(
        "Modèle de recommandation",
        ["Content-Based", "Constraint-Based", "Knowledge-Based", "Hybrid"]
    )
    top_k = st.sidebar.slider("Nombre de recommandations demandées", 3, 30, 10)

    # ------------------------------------------------------------------
    # Produit de référence (avec "(aucun)" possible)
    # ------------------------------------------------------------------
    st.markdown("### 🧾 Produit de référence")

    col_prod_select, col_prod_card = st.columns([1, 3])

    with col_prod_select:
        product_ids = products["Product ID"].tolist()
        options = ["(aucun)"] + [str(pid) for pid in product_ids]
        selected = st.selectbox("Product ID (pour Content-Based & Hybrid)", options)
        if selected == "(aucun)":
            base_product_id = None
        else:
            base_product_id = int(selected)

    with col_prod_card:
        if base_product_id is None:
            st.info(
                "Aucun produit de référence sélectionné. "
                "👉 Obligatoire pour **Content-Based** et **Hybrid**, "
                "facultatif pour **Constraint-Based** et **Knowledge-Based**."
            )
        else:
            base_product = products[products["Product ID"] == base_product_id].iloc[0]
            render_product_card(base_product)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Filtres au centre (constraint + knowledge)
    # ------------------------------------------------------------------
    st.markdown("### 🎛️ Filtres & Profil utilisateur")

    filter_col = st.container()
    with filter_col:
        col1, col2 = st.columns(2)

        # --- Constraint-Based Filters ---
        with col1:
            st.markdown("<div class='filter-box'>", unsafe_allow_html=True)
            st.markdown("**Filtres produit (Constraint-Based)**")
            use_constraints = st.checkbox("Activer les contraintes", value=False)

            category = brand = color = size = None
            max_price = min_price = None
            min_rating_constraint = None

            if use_constraints:
                category = st.selectbox(
                    "Category",
                    options=["(aucune)"] + sorted(products["Category"].unique().tolist())
                )
                if category == "(aucune)":
                    category = None

                brand = st.selectbox(
                    "Brand",
                    options=["(aucune)"] + sorted(products["Brand"].unique().tolist())
                )
                if brand == "(aucune)":
                    brand = None

                color = st.selectbox(
                    "Color",
                    options=["(aucune)"] + sorted(products["Color"].unique().tolist())
                )
                if color == "(aucune)":
                    color = None

                size = st.selectbox(
                    "Size",
                    options=["(aucune)"] + sorted(products["Size"].unique().tolist())
                )
                if size == "(aucune)":
                    size = None

                price_min, price_max = float(products["Price"].min()), float(products["Price"].max())
                price_range = st.slider(
                    "Price range",
                    float(price_min),
                    float(price_max),
                    (price_min, price_max)
                )
                min_price = price_range[0]
                max_price = price_range[1]

                min_rating_constraint = st.slider(
                    "Min rating (constraints)",
                    float(products["Rating"].min()),
                    float(products["Rating"].max()),
                    3.0,
                    step=0.1
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # --- Knowledge-Based Filters ---
        with col2:
            st.markdown("<div class='filter-box'>", unsafe_allow_html=True)
            st.markdown("**Profil utilisateur (Knowledge-Based)**")

            usage = st.selectbox(
                "Usage",
                options=["(aucun)", "sport", "casual", "work", "chic"]
            )
            if usage == "(aucun)":
                usage = None

            budget_level = st.selectbox(
                "Budget level",
                options=["(aucun)", "cheap", "mid", "premium"]
            )
            if budget_level == "(aucun)":
                budget_level = None

            preferred_color_kb = st.selectbox(
                "Preferred color",
                options=["(aucune)"] + sorted(products["Color"].unique().tolist())
            )
            if preferred_color_kb == "(aucune)":
                preferred_color_kb = None

            preferred_size_kb = st.selectbox(
                "Preferred size",
                options=["(aucune)"] + sorted(products["Size"].unique().tolist())
            )
            if preferred_size_kb == "(aucune)":
                preferred_size_kb = None

            preferred_brand_kb = st.selectbox(
                "Preferred brand",
                options=["(aucune)"] + sorted(products["Brand"].unique().tolist())
            )
            if preferred_brand_kb == "(aucune)":
                preferred_brand_kb = None

            min_rating_kb = st.slider(
                "Min rating (Knowledge-Based)",
                float(products["Rating"].min()),
                float(products["Rating"].max()),
                3.0,
                step=0.1
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Bouton + recommandations
    # ------------------------------------------------------------------
    center_btn_col = st.columns([3, 1, 3])[1]
    with center_btn_col:
        clicked = st.button("✨ Voir les recommandations")

    if clicked:
        st.markdown(f"### Résultats — Modèle : **{model_type}**")

        # Vérifications pour Content-Based & Hybrid
        if model_type in ["Content-Based", "Hybrid"] and base_product_id is None:
            st.error(
                "Le modèle **Content-Based** et le modèle **Hybrid** ont besoin "
                "d'un produit de référence. Merci d'en sélectionner un."
            )
            return

        # Calcul des recommandations
        if model_type == "Content-Based":
            recs = recommend_content_based(
                product_id=base_product_id,
                products=products,
                feature_matrix=feature_matrix,
                top_k=top_k
            )

        elif model_type == "Constraint-Based":
            recs = recommend_constraint_based(
                products=products,
                base_product_id=base_product_id if use_constraints else None,
                category=category if use_constraints else None,
                brand=brand if use_constraints else None,
                max_price=max_price if use_constraints else None,
                min_price=min_price if use_constraints else None,
                color=color if use_constraints else None,
                size=size if use_constraints else None,
                min_rating=min_rating_constraint if use_constraints else None,
                top_k=top_k
            )

        elif model_type == "Knowledge-Based":
            recs = knowledge_based_recommend(
                products=products,
                cheap_max=cheap_max,
                mid_max=mid_max,
                usage=usage,
                budget_level=budget_level,
                preferred_color=preferred_color_kb,
                preferred_size=preferred_size_kb,
                preferred_brand=preferred_brand_kb,
                min_rating=min_rating_kb,
                top_k=top_k
            )

        else:  # Hybrid
            recs = hybrid_recommend(
                base_product_id=base_product_id,
                products=products,
                feature_matrix=feature_matrix,
                cheap_max=cheap_max,
                mid_max=mid_max,
                category=category if use_constraints else None,
                brand=brand if use_constraints else None,
                max_price=max_price if use_constraints else None,
                min_price=min_price if use_constraints else None,
                color=color if use_constraints else None,
                size=size if use_constraints else None,
                min_rating_constraint=min_rating_constraint if use_constraints else None,
                usage=usage,
                budget_level=budget_level,
                preferred_color_kb=preferred_color_kb,
                preferred_size_kb=preferred_size_kb,
                preferred_brand_kb=preferred_brand_kb,
                min_rating_kb=min_rating_kb,
                # Poids fixes
                w_content=0.6,
                w_constraint=0.2,
                w_knowledge=0.2,
                top_k=top_k
            )

        if recs.empty:
            st.warning(
                "Aucun produit ne correspond aux filtres choisis. "
                "Essaye d’alléger les contraintes ou de baisser le min rating."
            )
        else:
            # Grille de cartes (3 par ligne)
            n_cols = 3
            recs = recs.reset_index(drop=True)
            rows = (len(recs) + n_cols - 1) // n_cols

            for i in range(rows):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i * n_cols + j
                    if idx < len(recs):
                        with cols[j]:
                            render_product_card(recs.loc[idx])

            # DataFrame brut optionnel
            with st.expander("Voir les données brutes (DataFrame)"):
                recs_display = recs.copy()
                if "Product ID" in recs_display.columns:
                    recs_display["Product ID"] = recs_display["Product ID"].astype(int)
                text_cols = ["Product Name", "Brand", "Category", "Color", "Size"]
                for col in text_cols:
                    if col in recs_display.columns:
                        recs_display[col] = recs_display[col].astype(str)
                st.dataframe(recs_display)


if __name__ == "__main__":
    main()
