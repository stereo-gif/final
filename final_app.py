import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, EnumerateStereoisomers
from stmol import showmol
import py3Dmol
import numpy as np

# ==============================
# 1. دالة الـ Ra/Sa للألين (مستقرة)
# ==============================
def get_allene_label(mol):
    try:
        m = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(m, AllChem.ETKDG()) == -1: return ""
        conf = m.GetConformer()
        for b in m.GetBonds():
            if b.GetBondType() == Chem.BondType.DOUBLE:
                a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
                for nb in a2.GetBonds():
                    if nb.GetIdx() == b.GetIdx(): continue
                    if nb.GetBondType() == Chem.BondType.DOUBLE:
                        a3 = nb.GetOtherAtom(a2)
                        l_subs = sorted([n for n in a1.GetNeighbors() if n.GetIdx()!=a2.GetIdx()], key=lambda x: x.GetAtomicNum(), reverse=True)
                        r_subs = sorted([n for n in a3.GetNeighbors() if n.GetIdx()!=a2.GetIdx()], key=lambda x: x.GetAtomicNum(), reverse=True)
                        if l_subs and r_subs:
                            p1, p3 = np.array(conf.GetAtomPosition(a1.GetIdx())), np.array(conf.GetAtomPosition(a3.GetIdx()))
                            pl, pr = np.array(conf.GetAtomPosition(l_subs[0].GetIdx())), np.array(conf.GetAtomPosition(r_subs[0].GetIdx()))
                            dot = np.dot(np.cross(pl-p1, p3-p1), pr-p3)
                            return "Ra" if dot > 0 else "Sa"
    except: return ""
    return ""

# ==============================
# 2. حل مشكلة الرسم الـ 2D (بدون Instantiation Error)
# ==============================
def draw_mol_2d(mol):
    # استخدام دالة Draw مباشرة لتجنب مشاكل الـ Abstract Class
    img = Draw.MolToImage(mol, size=(400, 400), kekulize=True, wedgeBonds=True)
    # wedgeBonds=True بتخلي الـ Stereochemistry (الروابط الطالعة والداخلة) واضحة جداً
    st.image(img, use_container_width=True)

# ==============================
# 3. دالة الـ 3D (اللي اشتغلت معاكي)
# ==============================
def render_3d(mol):
    m3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
    mblock = Chem.MolToMolBlock(m3d)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mblock, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    showmol(view, height=300, width=400)

# ==============================
# 4. التطبيق (UI)
# ==============================
st.set_page_config(layout="wide")
st.title("Final Stereo Analyzer (No-Error Edition)")

# المرجع العلمي اللي حفظناه
with st.expander("📚 Stereoisomerism Notes"):
    st.markdown("""
    - **Cis / Trans**: Relative side.
    - **E / Z**: Absolute priority side (Z = Same side).
    - **R / S**: Chiral center (R = Clockwise).
    - **Ra / Sa**: Axial chirality (Allenes).
    """)

name = st.text_input("Structure Name:", "2,3-pentadiene")

if st.button("Run Analysis"):
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            base_mol = Chem.MolFromSmiles(results[0].smiles)
            # توليد الأيزومرات
            opts = EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=True)
            isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(base_mol, options=opts))
            
            st.write(f"Found {len(isomers)} potential isomers.")
            
            for i, iso in enumerate(isomers):
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                axial = get_allene_label(iso)
                
                label = f"Isomer {i+1} | R/S: {centers}"
                if axial: label += f" | Axial: {axial}"
                
                st.subheader(label)
                col1, col2 = st.columns(2)
                with col1:
                    draw_mol_2d(iso)
                with col2:
                    render_3d(iso)
                st.divider()
        else:
            st.error("Not found.")
    except Exception as e:
        st.error(f"Error: {e}")
