import networkx as nx
from pyvis.network import Network
import streamlit as st
import tempfile

def plot_network(df, src_col="src_ip", dst_col="dst_ip", anomaly_col="anomaly_label"):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        src, dst, label = row[src_col], row[dst_col], row[anomaly_col]
        color = "red" if label == "Suspicious" else "blue"

        G.add_node(src, color="lightblue", title=f"Source: {src}")
        G.add_node(dst, color="lightgreen", title=f"Dest: {dst}")
        G.add_edge(src, dst, color=color)

    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)

    # Save to temp html and render in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        net.show(tmp.name)
        st.components.v1.html(open(tmp.name, "r", encoding="utf-8").read(), height=600)