import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, EnumerateStereoisomers
# استيراد rdMolDraw2D بطريقة أكثر أماناً
try:
    from rdkit.Chem.Draw import rdMolDraw2D
except ImportError:
    from rdkit.Chem import rdMolDraw2D

from stmol import showmol
import py3Dmol
import numpy as np

# ==============================
# 1. حساب الكايراليتي المحورية للألين
# ==============================
def get_allene_axial_config(mol):
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, AllChem.ETKDG()) == -1: return []
    conf = m.GetConformer()
    results = []
    for bond in m.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
            for nb in a2.GetBonds():
                if nb.GetIdx() == bond.GetIdx(): continue
                if nb.GetBondType() == Chem.BondType.DOUBLE:
                    a3 = nb.GetOtherAtom(a2)
                    l_subs = sorted([n for n in a1.GetNeighbors() if n.GetIdx()!=a2.GetIdx()], key=lambda x: x.GetAtomicNum(), reverse=True)
                    r_subs = sorted([n for n in a3.GetNeighbors() if n.GetIdx()!=a2.GetIdx()], key=lambda x: x.GetAtomicNum(), reverse=True)
                    if len(l_subs) >= 2 and len(r_subs) >= 2:
                        p1, p3 = np.array(conf.GetAtomPosition(a1.GetIdx())), np.array(conf.GetAtomPosition(a3.GetIdx()))
                        pl, pr = np.array(conf.GetAtomPosition(l_subs[0].GetIdx())), np.array(conf.GetAtomPosition(r_subs[0].GetIdx()))
                        dot = np.dot(np.cross(pl-p1, p3-p1), pr-p3)
                        results.append("Ra" if dot > 0 else "Sa")
    return results

# ==============================
# 2. حل مشكلة الرسم (Pretty 2D)
# ==============================
def render_pretty_2d(mol, label, axial_label=""):
    mc = Chem.Mol(mol)
    AllChem.Compute2DCoords(mc)
    
    # محاولة استخدام السحب (SVG) بطريقة متوافقة
    try:
        drawer = rdMolDraw2D.MolDraw2DSvg(400, 400)
    except AttributeError:
        # حل بديل لو النسخة مختلفة
        drawer = rdMolDraw2D.MolDraw2D(400, 400)
        
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.atomLabelFontSize = 25
    options.bondLineWidth = 3
    
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    st.write(f"### {label}")
    if axial_label:
        st.success(f"**Axial Chirality:** {axial_label}")
    
    # عرض الـ SVG مباشرة في Streamlit
    st.image(svg, use_container_width=True)

# ==============================
# 3. عرض الـ 3D (المستقر)
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
# 4. واجهة البرنامج الرئيسية
# ==============================
st.set_page_config(page_title="StereoMaster 2026", layout="wide")

# النوت العلمية اللي طلبتيها
with st.sidebar:
    st.header("📚 Stereoisomerism Guide")
    st.info("""
    - **R / S**: Chiral center configuration.
    - **Ra / Sa**: Axial chirality in Allenes.
    - **E / Z**: Absolute double bond geometry.
    """)

name = st.text_input("Enter Molecule Name:", "2,3-pentadiene")

if st.button("Generate Isomers"):
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            base_mol = Chem.MolFromSmiles(results[0].smiles)
            opts = EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(base_mol, options=opts))
            
            st.write(f"Found {len(isomers)} possible isomers.")
            
            for i, iso in enumerate(isomers):
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                axial = get_allene_axial_config(iso)
                
                col1, col2 = st.columns(2)
                with col1:
                    render_pretty_2d(iso, f"Isomer {i+1} (R/S: {centers})", axial_label=axial)
                with col2:
                    st.write("**3D Interactive Model**")
                    render_3d(iso)
                st.divider()
        else:
            st.error("Compound not found.")
    except Exception as e:
        st.error(f"Error: {e}")
