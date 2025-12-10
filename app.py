import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Konfigurasi halaman
st.set_page_config(
    page_title="Grocery Market Basket Analysis",
    page_icon="🛒",
    layout="wide"
)

# Judul aplikasi
st.title("🛒 Grocery Market Basket Analysis")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 Dashboard", "🔍 Product Analysis", "💡 Recommendations", "📈 Association Rules"]
)

# Load data
@st.cache_data
def load_data():
    # Load transactions
    df = pd.read_csv('groceries.csv')
    transactions = []
    for i in range(len(df)):
        trans = [str(item).strip() for item in df.iloc[i].tolist()
                if pd.notna(item) and str(item).strip() != '']
        transactions.append(trans[1:])

    # Load association rules
    rules = pd.read_csv('association_rules.csv')

    # Hitung item frequencies
    all_items = [item for transaction in transactions for item in transaction]
    item_freq = pd.Series(all_items).value_counts().reset_index()
    item_freq.columns = ['item', 'frequency']

    return transactions, rules, item_freq

transactions, rules, item_freq = load_data()

if page == "📊 Dashboard":
    st.header("📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", len(transactions))

    with col2:
        unique_items = len(item_freq)
        st.metric("Unique Products", unique_items)

    with col3:
        avg_items = np.mean([len(t) for t in transactions])
        st.metric("Avg Items per Transaction", f"{avg_items:.2f}")

    # Transaction length distribution
    st.subheader("Transaction Size Distribution")
    trans_lengths = [len(t) for t in transactions]

    fig1 = px.histogram(
        x=trans_lengths,
        nbins=30,
        title="Distribution of Items per Transaction",
        labels={'x': 'Number of Items', 'y': 'Frequency'},
        color_discrete_sequence=['#2E86AB']
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Top products
    st.subheader("Top 20 Most Popular Products")
    fig2 = px.bar(
        item_freq.head(20),
        x='frequency',
        y='item',
        orientation='h',
        title="Most Frequently Purchased Items",
        color='frequency',
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🔍 Product Analysis":
    st.header("🔍 Product Analysis")

    # Pilih produk untuk dianalisis
    selected_product = st.selectbox(
        "Select a product to analyze",
        item_freq['item'].head(50).tolist()
    )

    if selected_product:
        # Hitung co-occurrence
        co_occurrences = Counter()
        for transaction in transactions:
            if selected_product in transaction:
                for item in transaction:
                    if item != selected_product:
                        co_occurrences[item] += 1

        # Buat DataFrame co-occurrence
        co_occur_df = pd.DataFrame(
            co_occurrences.items(),
            columns=['product', 'co_occurrence_count']
        ).sort_values('co_occurrence_count', ascending=False).head(20)

        st.subheader(f"Products Frequently Bought With {selected_product}")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                co_occur_df,
                x='co_occurrence_count',
                y='product',
                orientation='h',
                title=f"Top Products Bought with {selected_product}",
                color='co_occurrence_count',
                color_continuous_scale='Plasma'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric(
                "Transactions with this product",
                sum(1 for t in transactions if selected_product in t)
            )
            st.metric(
                "Total occurrences",
                item_freq[item_freq['item'] == selected_product]['frequency'].values[0]
            )

        # Show transactions containing this product
        st.subheader("Sample Transactions")
        sample_trans = [t for t in transactions if selected_product in t][:10]
        for i, trans in enumerate(sample_trans, 1):
            st.write(f"{i}. {', '.join(trans)}")

elif page == "💡 Recommendations":
    st.header("💡 Product Recommendations")

    # Multi-select untuk memilih produk
    selected_products = st.multiselect(
        "Select products in cart",
        item_freq['item'].head(100).tolist(),
        max_selections=5
    )

    if selected_products:
        st.subheader("Recommended Products to Add:")

        # Cari rules yang relevan
        recommendations = {}

        for idx, row in rules.iterrows():
            antecedents = eval(row['antecedents']) if isinstance(row['antecedents'], str) else row['antecedents']
            consequents = eval(row['consequents']) if isinstance(row['consequents'], str) else row['consequents']

            # Cek jika antecedents subset dari selected_products
            if all(item in selected_products for item in antecedents):
                for consequent in consequents:
                    if consequent not in selected_products:
                        if consequent not in recommendations:
                            recommendations[consequent] = {
                                'confidence': row['confidence'],
                                'lift': row['lift'],
                                'support': row['support']
                            }
                        else:
                            # Update dengan rule terbaik
                            if row['confidence'] > recommendations[consequent]['confidence']:
                                recommendations[consequent] = {
                                    'confidence': row['confidence'],
                                    'lift': row['lift'],
                                    'support': row['support']
                                }

        if recommendations:
            # Urutkan berdasarkan confidence
            rec_df = pd.DataFrame([
                {
                    'product': product,
                    'confidence': data['confidence'],
                    'lift': data['lift'],
                    'support': data['support']
                }
                for product, data in recommendations.items()
            ]).sort_values('confidence', ascending=False).head(10)

            # Tampilkan rekomendasi
            for idx, row in rec_df.iterrows():
                with st.expander(f"➕ {row['product']}"):
                    st.metric("Confidence", f"{row['confidence']:.2%}")
                    st.metric("Lift", f"{row['lift']:.2f}")
                    st.metric("Support", f"{row['support']:.3f}")

                    # Tampilkan rules terkait
                    related_rules = rules[
                        rules['consequents'].apply(
                            lambda x: row['product'] in (eval(x) if isinstance(x, str) else x)
                        )
                    ].head(3)

                    for _, rule in related_rules.iterrows():
                        antecedents = eval(rule['antecedents']) if isinstance(rule['antecedents'], str) else rule['antecedents']
                        st.write(f"If {', '.join(antecedents)} → Then {row['product']}")
                        st.write(f"Confidence: {rule['confidence']:.2%}, Lift: {rule['lift']:.2f}")
                        st.write("---")
        else:
            st.info("No specific recommendations found. Try adding more products to your cart.")

    else:
        st.info("Select products from your shopping cart to get recommendations.")

elif page == "📈 Association Rules":
    st.header("📈 Association Rules Analysis")

    # Filter rules
    col1, col2, col3 = st.columns(3)

    with col1:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )

    with col2:
        min_lift = st.slider(
            "Minimum Lift",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1
        )

    with col3:
        min_support = st.slider(
            "Minimum Support",
            min_value=0.0,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f"
        )

    # Filter rules
    filtered_rules = rules[
        (rules['confidence'] >= min_confidence) &
        (rules['lift'] >= min_lift) &
        (rules['support'] >= min_support)
    ].sort_values('lift', ascending=False)

    st.subheader(f"Found {len(filtered_rules)} Rules")

    # Tampilkan rules
    for idx, row in filtered_rules.head(50).iterrows():
        antecedents = eval(row['antecedents']) if isinstance(row['antecedents'], str) else row['antecedents']
        consequents = eval(row['consequents']) if isinstance(row['consequents'], str) else row['consequents']

        with st.expander(f"{', '.join(antecedents)} → {', '.join(consequents)}"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Support", f"{row['support']:.3f}")

            with col2:
                st.metric("Confidence", f"{row['confidence']:.2%}")

            with col3:
                st.metric("Lift", f"{row['lift']:.2f}")

            with col4:
                st.metric("Conviction", f"{row['conviction']:.2f}")

    # Visualisasi rules
    if len(filtered_rules) > 0:
        st.subheader("Rules Visualization")

        fig = px.scatter(
            filtered_rules.head(50),
            x='support',
            y='confidence',
            size='lift',
            color='lift',
            hover_data=['antecedents', 'consequents'],
            title='Association Rules (Support vs Confidence)',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Market Basket Analysis using Apriori Algorithm</p>
        <p>Data Source: Groceries Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)
