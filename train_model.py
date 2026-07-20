"""
Huấn luyện mô hình DILI demo: MACCS 166 Keys + Random Forest
(theo phương pháp tốt nhất trong khóa luận Phan Chí Hải, mục 3.5.2.1)

LƯU Ý: dùng bộ dữ liệu DEMO nhỏ (~40 chất) chỉ để minh họa quy trình web app.
Để có mô hình dùng thật, hãy thay demo_data bằng dữ liệu DILIrank/LiverTox/DILI-DB
thật (xem notebook DILI_MXAINet_Reconstructed.ipynb, mục 1) rồi chạy lại file này.
"""
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42

# Bộ dữ liệu DEMO (giống notebook DILI_MXAINet_Reconstructed.ipynb)
demo_data = [
    ("CC(=O)Nc1ccc(O)cc1", 1),
    ("CC(=O)Oc1ccccc1C(=O)O", 0),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", 0),
    ("Clc1ccc2c(c1)C(=O)c1ccccc1N2", 0),
    ("CN1CCC[C@H]1c1cccnc1", 0),
    ("CC(=O)Nc1ccc(cc1)S(N)(=O)=O", 0),
    ("O=C(Nc1ccc(cc1)N1CCOCC1)c1ccccc1", 0),
    ("Clc1ccccc1C1=NCC(=O)Nc2ccccc21", 1),
    ("CC1=C(C(=NO1)C2=CC=CC=C2Cl)C(=O)O", 0),
    ("CCN(CC)CCNC(=O)c1cc(Cl)c(N)cc1OC", 0),
    ("CC(C)NCC(O)COc1ccc2[nH]ccc2c1", 0),
    ("CN(C)CCC=C1c2ccccc2CCc2ccccc21", 0),
    ("O=C1CCC(=O)N1SC(Cl)(Cl)Cl", 1),
    ("CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2O", 0),
    ("CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", 0),
    ("Nc1ccn(C2OC(CO)C(O)C2O)c(=O)n1", 0),
    ("Cc1onc(-c2c(F)cccc2Cl)c1C(=O)NC1C(=O)N2C1SC(C)(C)C2C(=O)O", 1),
    ("CC(C)Cc1nc(no1)C(=O)NCC1CCC(CC1)C(=O)O", 0),
    ("Clc1ccc(cc1)C(c1ccccc1Cl)N1CCN(CC1)CCOCCO", 0),
    ("CCOC(=O)C1=C(C)NC(C)=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC", 1),
    ("CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc12", 1),
    ("CC(C)NCC(COc1cccc2ccccc12)O", 0),
    ("CN1CCC23c4c1cc1ccccc1c4C(=O)CCC2N(C)CC3", 1),
    ("COc1cc2c(cc1OC)C(=O)C(CC1CCN(C)CC1)C2", 0),
    ("O=C(O)c1ccccc1Nc1c(Cl)cccc1Cl", 1),
    ("Cc1ccc(cc1)S(=O)(=O)Nc1ccc(Cl)cc1", 0),
    ("Nc1ccc(cc1)S(N)(=O)=O", 0),
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", 0),
    ("CC(=O)Oc1ccccc1", 0),
    ("O=C1c2ccccc2C(=O)c2c1ccc1c2ccc2c1cccc2", 0),
    ("Clc1cc2c(cc1Cl)Oc1cc(Cl)c(Cl)cc1O2", 1),
    ("CC(C)Nc1ncnc2c1ncn2C1OC(CO)C(O)C1O", 0),
    ("CCCCCCCCCCCCCCCCCC(=O)O", 0),
    ("COc1cc(cc(OC)c1O)C(=O)Nc1ccc(cc1)S(=O)(=O)N1CCCCC1", 1),
    ("Cc1cc(C)c(NC(=O)COc2ccc(Cl)cc2Cl)c(C)c1", 1),
    ("Oc1ccc(cc1)C(=O)c1ccc(O)cc1", 0),
    ("CC(C)(C)NCC(O)COc1cccc2ccccc12", 0),
    ("CN1CCN(CC1)c1ccc(cc1)Nc1nc2ccccc2n1", 1),
    ("Clc1ccc2c(c1)nc1ccccc1c2=NNc1ccccc1", 1),
    ("CC(C)NCC(O)c1ccc(O)c(O)c1", 0),
    ("CCC1(C(=O)NC(=O)NC1=O)c1ccccc1", 0),
    ("Cc1ccc2nc3ccccc3c(=O)n2n1", 1),
    ("O=C(Cn1ccnc1[N+](=O)[O-])NCCO", 0),
    ("CC(C)Cc1ccccc1", 0),
    ("Clc1ccccc1-c1nnc(SCC(=O)Nc2ccccc2)n1N", 1),
]


def featurize(smiles_list):
    feats = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        feats.append(list(MACCSkeys.GenMACCSKeys(mol)))
        valid_idx.append(i)
    return np.array(feats), valid_idx


def main():
    smiles = [s for s, _ in demo_data]
    labels = np.array([l for _, l in demo_data])

    X, valid_idx = featurize(smiles)
    y = labels[valid_idx]

    model = RandomForestClassifier(
        n_estimators=300, max_depth=20, max_features="sqrt",
        min_samples_split=5, criterion="gini", random_state=RANDOM_STATE,
    )
    model.fit(X, y)

    with open("model/dili_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Đã huấn luyện xong trên {len(y)} mẫu demo. Model lưu tại model/dili_model.pkl")


if __name__ == "__main__":
    main()
