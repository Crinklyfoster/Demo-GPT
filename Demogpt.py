import streamlit as st
import psycopg2
from rdkit import Chem
from rdkit.Chem import Draw
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch_geometric.nn import GCNConv
import torch
from torch_geometric.data import Data
import pandas as pd

# Load Falcon GPT Model
@st.cache_resource
def load_gpt_model():
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct")
    return tokenizer, model

tokenizer, gpt_model = load_gpt_model()

# Database connection
@st.cache_resource
def connect_db():
    return psycopg2.connect(
        host="localhost",
        database="chemistry_db",
        user="your_user",
        password="your_password"
    )

# GNN Model Definition
class MoleculeGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(MoleculeGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Predict one property (e.g., logP)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.fc(x)

# Pre-trained GNN Model Loader
@st.cache_resource
def load_gnn_model():
    model = MoleculeGNN(num_features=1, hidden_channels=128)
    model.load_state_dict(torch.load("gnn_model.pth"))  # Path to the pre-trained model
    model.eval()
    return model

gnn_model = load_gnn_model()

# Convert SMILES to Graph
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# GPT Query Generation
def generate_query(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = gpt_model.generate(inputs, max_length=200, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Execute SQL Query
def execute_query(query):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return results
    except Exception as e:
        return f"Error: {e}"

# Streamlit App
st.title("Chemical Data Explorer - GNN and GPT Powered")
st.markdown("Search compounds, predict properties, and explore chemical data.")

# Tabs for Functionality
tab1, tab2, tab3 = st.tabs(["Natural Language Query", "SMILES Query", "Predict with GNN"])

# Tab 1: Natural Language Query
with tab1:
    st.subheader("Ask a natural language question")
    natural_query = st.text_area("Example: 'Find compounds with molecular weight > 200'")
    if st.button("Run GPT Query"):
        if natural_query.strip():
            with st.spinner("Processing with Falcon GPT..."):
                gpt_query = generate_query(f"Convert this to SQL query: {natural_query}")
                st.write("Generated SQL Query:")
                st.code(gpt_query)
                results = execute_query(gpt_query)
                if isinstance(results, list):
                    st.success("Query executed successfully.")
                    st.dataframe(results)
                else:
                    st.error(results)
        else:
            st.error("Please enter a query.")

# Tab 2: SMILES Query
with tab2:
    st.subheader("Search by SMILES")
    smiles_input = st.text_input("Enter a SMILES string")
    if st.button("Search by SMILES"):
        if smiles_input.strip():
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="Molecular Structure", use_column_width=True)
                query = f"""
                SELECT id, name, smiles, formula, molecular_weight
                FROM compounds
                WHERE mol @> '{Chem.MolToSmiles(mol)}'::qmol
                """
                results = execute_query(query)
                if isinstance(results, list):
                    st.success("Query executed successfully.")
                    st.dataframe(results)
                else:
                    st.error(results)
            else:
                st.error("Invalid SMILES string.")
        else:
            st.error("Please enter a SMILES string.")

# Tab 3: Predict with GNN
with tab3:
    st.subheader("Predict molecular properties using GNN")
    smiles_input = st.text_input("Enter a SMILES for GNN Prediction")
    if st.button("Predict Property"):
        if smiles_input.strip():
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                graph = smiles_to_graph(smiles_input)
                with torch.no_grad():
                    prediction = gnn_model(graph.x, graph.edge_index).item()
                st.success(f"Predicted property: {prediction:.3f}")
            else:
                st.error("Invalid SMILES string.")
        else:
            st.error("Please enter a SMILES string.")

# Footer
st.markdown("---")
st.caption("Powered by Streamlit, Falcon GPT, RDKit, and PyTorch Geometric")
