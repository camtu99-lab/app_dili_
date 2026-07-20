"""
Web app demo: Dự đoán nguy cơ tổn thương gan do thuốc (DILI) từ SMILES
Theo mục 3.6 khóa luận Phan Chí Hải — bản dựng lại bằng Flask.

Chạy:
    pip install -r requirements.txt
    python train_model.py     # huấn luyện mô hình demo (chỉ cần chạy 1 lần)
    python app.py
    Mở trình duyệt tại http://127.0.0.1:5000
"""
import os
import pickle

import numpy as np
from flask import Flask, render_template, request
from rdkit import Chem
from rdkit.Chem import Draw, MACCSkeys
from rdkit.Chem.Draw import rdMolDraw2D

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model", "dili_model.pkl")

app = Flask(__name__)

_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Chưa có model/dili_model.pkl — hãy chạy `python train_model.py` trước."
            )
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def smiles_to_mol_svg(mol, size=300):
    drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    smiles_input = ""
    mol_svg = None

    if request.method == "POST":
        smiles_input = request.form.get("smiles", "").strip()
        mol = Chem.MolFromSmiles(smiles_input) if smiles_input else None

        if not smiles_input:
            error = "Vui lòng nhập chuỗi SMILES."
        elif mol is None:
            error = f"SMILES không hợp lệ: '{smiles_input}'. Vui lòng kiểm tra lại."
        else:
            try:
                model = get_model()
                features = np.array([list(MACCSkeys.GenMACCSKeys(mol))])
                pred = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                mol_svg = smiles_to_mol_svg(mol)
                result = {
                    "label": "DILI-positive (Có nguy cơ)" if pred == 1 else "DILI-negative (Không có nguy cơ)",
                    "is_positive": bool(pred == 1),
                    "prob_positive": round(float(proba[1]) * 100, 1),
                    "prob_negative": round(float(proba[0]) * 100, 1),
                    "canonical_smiles": Chem.MolToSmiles(mol),
                }
            except FileNotFoundError as e:
                error = str(e)

    return render_template(
        "index.html",
        result=result,
        error=error,
        smiles_input=smiles_input,
        mol_svg=mol_svg,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
