"""
Web app demo: Dự đoán nguy cơ tổn thương gan do thuốc (DILI) từ SMILES
Bản Streamlit — dựng lại theo mục 3.6 khóa luận Phan Chí Hải.

Chạy local:
    pip install -r requirements.txt
    python train_model.py     # huấn luyện mô hình demo (chỉ cần 1 lần)
    streamlit run streamlit_app.py

Deploy public miễn phí:
    Đẩy repo này lên GitHub -> vào https://share.streamlit.io -> "New app"
    -> chọn repo -> chọn file "streamlit_app.py" -> Deploy.
"""
import os
import pickle

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "dili_model.pkl")

st.set_page_config(page_title="DILI Prediction Demo", page_icon="🧪", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def mol_to_svg(mol, size=280):
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 20px;">
        <h1 style="margin-bottom:0;">🧪 DILI Prediction Demo</h1>
        <p style="color:#6b7d75; margin-top:4px;">
            Dự đoán nguy cơ tổn thương gan do thuốc (Drug-Induced Liver Injury) từ cấu trúc SMILES
        </p>
        <p style="color:#9aa8a1; font-size:0.82rem;">
            Bản dựng lại minh họa dựa trên khóa luận tốt nghiệp — mô hình MACCS 166 Keys + Random Forest
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()
if model is None:
    st.error("Chưa có `model/dili_model.pkl` — hãy chạy `python train_model.py` trước rồi khởi động lại app.")
    st.stop()

example_cols = st.columns(4)
example_smiles = {
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Diclofenac": "O=C(O)c1ccccc1Nc1c(Cl)cccc1Cl",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}

if "smiles_value" not in st.session_state:
    st.session_state.smiles_value = ""

for col, (name, smi) in zip(example_cols, example_smiles.items()):
    if col.button(name, use_container_width=True):
        st.session_state.smiles_value = smi

smiles_input = st.text_input(
    "Nhập chuỗi SMILES của hợp chất:",
    value=st.session_state.smiles_value,
    placeholder="Ví dụ: CC(=O)Nc1ccc(O)cc1  (Paracetamol)",
    key="smiles_box",
)

predict_clicked = st.button("Dự đoán", type="primary")

if predict_clicked:
    if not smiles_input.strip():
        st.warning("Vui lòng nhập chuỗi SMILES.")
    else:
        mol = Chem.MolFromSmiles(smiles_input.strip())
        if mol is None:
            st.error(f"SMILES không hợp lệ: '{smiles_input}'. Vui lòng kiểm tra lại.")
        else:
            features = np.array([list(MACCSkeys.GenMACCSKeys(mol))])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            col_img, col_result = st.columns([1, 1.4])
            with col_img:
                st.image(mol_to_svg(mol), use_container_width=True)
            with col_result:
                if pred == 1:
                    st.markdown("### 🔴 DILI-positive (Có nguy cơ)")
                else:
                    st.markdown("### 🟢 DILI-negative (Không có nguy cơ)")

                st.write(f"**Xác suất DILI-positive:** {proba[1]*100:.1f}%")
                st.progress(float(proba[1]))
                st.write(f"**Xác suất DILI-negative:** {proba[0]*100:.1f}%")
                st.progress(float(proba[0]))

                st.caption(f"Canonical SMILES: `{Chem.MolToSmiles(mol)}`")

st.markdown("---")
st.caption(
    "⚠️ **Lưu ý:** đây là bản demo minh họa quy trình, mô hình được huấn luyện trên bộ dữ liệu mẫu rất nhỏ "
    "(~40 chất), **không phải** mô hình đã huấn luyện trên bộ dữ liệu thật (DILIrank/LiverTox/DILI-DB, "
    "~5.172 mẫu) trong khóa luận. Kết quả dự đoán ở đây chỉ mang tính chất kỹ thuật, không dùng cho mục "
    "đích sàng lọc dược phẩm thực tế."
)
