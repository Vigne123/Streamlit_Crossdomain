# ---------------- Streamlit App ----------------
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html

st.set_page_config(layout="wide")
st.title("News Knowledge Graph Dashboard")

# Load data
try:
    df = pd.read_csv("news_dataset_cleaned.csv")
    triples = pd.read_csv("meaningful_triples_cleaned.csv")
except FileNotFoundError:
    st.error("Please run preprocessing and triple extraction first.")
    st.stop()

# ---------------- Dataset Overview ----------------
st.subheader("Dataset Overview")
st.dataframe(df.head(20))

# ---------------- Category Distribution ----------------
st.subheader("Category Distribution")
if "category" in df.columns:
    category_counts = df['category'].value_counts()
    st.bar_chart(category_counts)
else:
    st.warning("No 'category' column found in dataset.")

# ---------------- Triple Extraction ----------------
st.subheader("Top Extracted Triples")
st.dataframe(triples.head(20))

# ---------------- Semantic Search ----------------
st.subheader("Semantic Search: View Related Triples in Graph")

# Single keyword search (works across entities, relation, and category)
search_term = st.text_input("Enter keyword (entity, relation, or category):")

if search_term:
    filtered = triples[
        triples[['Entity1', 'Relation', 'Entity2', 'Category']]
        .astype(str)
        .apply(lambda row: row.str.contains(search_term, case=False)).any(axis=1)
    ]
    st.dataframe(filtered)

    # Build graph from filtered triples
    G = nx.DiGraph()
    for _, row in filtered.iterrows():
        subj, rel, obj = row['Entity1'], row['Relation'], row['Entity2']
        G.add_node(subj, color='skyblue', title=subj)
        G.add_node(obj, color='lightgreen', title=obj)
        G.add_edge(subj, obj, title=rel, label=rel)

    if len(G.nodes) > 0:
        net = Network(height="600px", width="100%", directed=True)
        net.from_nx(G)
        net.repulsion(node_distance=200, central_gravity=0.2)
        net.show_buttons(filter_=['physics'])
        html_file = "search_triples_graph.html"
        net.save_graph(html_file)
        with open(html_file, "r", encoding="utf-8") as f:
            graph_html = f.read()
        html(graph_html, height=600)
    else:
        st.info("No triples found for this keyword.")
