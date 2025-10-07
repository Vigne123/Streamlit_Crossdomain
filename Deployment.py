import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
from sentence_transformers import SentenceTransformer, util
from streamlit.components.v1 import html
import matplotlib.pyplot as plt
import random

# -------------------- Load Triples --------------------
@st.cache_data
def load_triples(path="meaningful_triples_cleaned.csv"):
    return pd.read_csv(path)

# -------------------- Build Graph --------------------
def build_graph(triples_df):
    G = nx.DiGraph()
    categories = list(triples_df['Category'].dropna().unique())
    cat_colors = {cat: "#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for cat in categories}

    for _, row in triples_df.iterrows():
        e1, rel, e2, cat = str(row["Entity1"]), str(row["Relation"]), str(row["Entity2"]), row["Category"]
        G.add_node(e1, category=cat, color=cat_colors.get(cat, "#CCCCCC"))
        G.add_node(e2, category=cat, color=cat_colors.get(cat, "#CCCCCC"))
        G.add_edge(e1, e2, relation=rel)
    return G, cat_colors

# -------------------- Graph Analytics --------------------
def compute_centralities(G):
    und = G.to_undirected()
    return {
        "degree": dict(nx.degree(G)),
        "betweenness": nx.betweenness_centrality(und),
        "closeness": nx.closeness_centrality(und),
    }

def detect_communities(G):
    und = G.to_undirected()
    comms = nx.algorithms.community.greedy_modularity_communities(und)
    return [list(c) for c in comms]

def plot_centrality_bar(data_dict, title, top_n=10):
    sorted_nodes = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    nodes, scores = zip(*sorted_nodes)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(nodes[::-1], scores[::-1], color="skyblue")
    ax.set_title(f"Top {top_n} by {title}", fontsize=14)
    ax.set_xlabel("Score")
    st.pyplot(fig)

def pyvis_network(G, height="600px"):
    net = Network(height=height, width="100%", directed=True)
    for u, v, d in G.edges(data=True):
        u_color = G.nodes[u].get('color', '#CCCCCC')
        v_color = G.nodes[v].get('color', '#CCCCCC')
        net.add_node(u, label=u, color=u_color)
        net.add_node(v, label=v, color=v_color)
        net.add_edge(u, v, label=d.get('relation', ''))
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])
    net.write_html("graph_analytics.html")
    return "graph_analytics.html"

# -------------------- Semantic QA --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def semantic_search(triples_df, question, model, top_k=1):
    triple_sentences = triples_df.apply(
        lambda row: f"{row['Entity1']} {row['Relation']} {row['Entity2']}", axis=1
    ).tolist()
    triple_embeddings = model.encode(triple_sentences, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)
    cos_scores = util.cos_sim(question_embedding, triple_embeddings)[0]
    top_results = cos_scores.topk(k=top_k)
    results = [(triple_sentences[idx], float(cos_scores[idx])) for idx in top_results[1]]
    return results

def visualize_node_graph(G, entity):
    if entity not in G:
        st.warning(f"Entity '{entity}' not found in the graph.")
        return None
    net = Network(height="600px", width="100%", directed=True)
    net.add_node(entity, label=entity, color=G.nodes[entity].get('color', 'green'))
    
    for nbr in G.successors(entity):
        rel = G.edges[entity, nbr]["relation"]
        net.add_node(nbr, label=nbr, color=G.nodes[nbr].get('color', '#CCCCCC'))
        net.add_edge(entity, nbr, label=rel)
    for nbr in G.predecessors(entity):
        rel = G.edges[nbr, entity]["relation"]
        net.add_node(nbr, label=nbr, color=G.nodes[nbr].get('color', '#CCCCCC'))
        net.add_edge(nbr, entity, label=rel)
    
    net.show_buttons(filter_=['physics'])
    net.write_html("semantic_node_graph.html")
    return "semantic_node_graph.html"

# -------------------- Streamlit UI --------------------
st.title("Graph Analytics + Semantic QA with Cross-Domain Linking")

# Load data and graph
triples_df = load_triples()
G, cat_colors = build_graph(triples_df)

# --- Graph Analytics ---
st.subheader("Graph Analytics")
if st.button("Run Graph Analytics"):
    st.write("### Centralities")
    centralities = compute_centralities(G)
    for name, scores in centralities.items():
        st.write(f"#### {name.capitalize()} Centrality")
        df = pd.DataFrame(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10],
                          columns=["Node", "Score"])
        st.table(df)
        plot_centrality_bar(scores, name)
    
    st.write("### Communities")
    communities = detect_communities(G)
    st.write(f"Detected {len(communities)} communities")
    st.write("First 5 communities:", communities[:5])
    
    st.write("### Graph Visualization")
    graph_file = pyvis_network(G)
    with open(graph_file, "r", encoding="utf-8") as f:
        html(f.read(), height=650)

# --- Semantic QA ---
st.subheader("Semantic QA over Knowledge Graph")
model = load_model()
question = st.text_input("Ask a question about entities:")
if st.button("Search QA"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        results = semantic_search(triples_df, question, model, top_k=1)
        if not results:
            st.info("No matching triples found.")
        else:
            st.write("### Top Matching Triple")
            for triple, score in results:
                st.write(f"{triple}  (Score: {score:.3f})")
            top_triple = results[0][0]
            entity1 = triples_df[triples_df.apply(
                lambda row: f"{row['Entity1']} {row['Relation']} {row['Entity2']}" == top_triple,
                axis=1
            )].iloc[0]['Entity1']
            
            # Show category legend for this entity's category
            entity_cat = G.nodes[entity1].get('category', None)
            if entity_cat:
                st.write("### Category Legend for This Result")
                color = G.nodes[entity1].get('color', '#CCCCCC')
                st.markdown(f"<span style='display:inline-block;width:20px;height:20px;background:{color};margin-right:5px;'></span>{entity_cat}", unsafe_allow_html=True)
            
            # Node-level graph
            st.write(f"### Graph Visualization for: {entity1}")
            graph_file = visualize_node_graph(G, entity1)
            if graph_file:
                with open(graph_file, "r", encoding="utf-8") as f:

                    st.components.v1.html(f.read(), height=600)
