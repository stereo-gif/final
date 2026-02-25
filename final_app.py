import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Advanced Chemical Isomer Analysis", layout="wide")

# 2. تصميم الواجهة
st.markdown("""
<style>
    .stApp { background-color: white; color: black; }
    .reportview-container { background: white; }
</style>
<h2 style='color: #800000; font-family: serif; border-bottom: 2px solid #dcdde1;'>Chemical Isomer Analysis System 2.0</h2>
<div style="background-color: #f9f9f9; padding: 15px; border: 1px solid #e1e1e1; border-left: 4px solid #800000; margin-bottom: 20px; font-family: sans-serif;">
    <strong style="color: #800000;">Stereoisomerism Reference Guide:</strong><br>
    1. <b style="color: #b22222;">Cis / Trans:</b> Identical groups on same/opposite sides.<br>
    2. <b style="color: #b22222;">E / Z (CIP System):</b> <b>Z (Zusammen)</b> together, <b>E (Entgegen)</b> opposite.<br>
    3. <b style="color: #b22222;">R / S (Optical):</b> Absolute configuration of chiral centers.<br>
    4. <b style="color: #b22222;">Ra / Sa (Axial):</b> Axial chirality in Allenes (C=C=C).
</div>
""", unsafe_allow_html=True)

# دالة حساب أيزومرات الألين Ra/Sa
def get_allene_stereo(mol):
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
                    if len(l_subs) >= 1 and len(r_subs) >= 1:
                        p1, p3 = np.array(conf.GetAtomPosition(a1.GetIdx())), np.array(conf.GetAtomPosition(a3.GetIdx()))
                        pl, pr = np.array(conf.GetAtomPosition(l_subs[0].GetIdx())), np.array(conf.GetAtomPosition(r_subs[0].GetIdx()))
                        dot = np.dot(np.cross(pl-p1, p3-p1), pr-p3)
                        results.append("Ra" if dot > 0 else "Sa")
    return results

def render_3d(mol, title):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    mblock = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mblock, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    st.write(f"**{title}**")
    showmol(view, height=300, width=400)

compound_name = st.text_input("Enter Structure Name (e.g., 2,3-pentadiene):", "")

if st.button("Analyze & Visualize Isomers"):
    if not compound_name:
        st.warning("Please enter a compound name first.")
    else:
        try:
            results = pcp.get_compounds(compound_name, 'name')
            if not results:
                st.error(f"❌ No compound found for: {compound_name}")
            else:
                base_smiles = results[0].smiles
                mol = Chem.MolFromSmiles(base_smiles)
                
                mol_no_stereo = Chem.Mol(mol)
                for bond in mol_no_stereo.GetBonds(): bond.SetStereo(Chem.BondStereo.STEREONONE)
                for atom in mol_no_stereo.GetAtoms(): atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                
                # --- التعديل الجوهري هنا ---
                opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
                isomers = list(EnumerateStereoisomers(mol_no_stereo, options=opts))
                
                st.subheader("1. Isomeric Relationships")
                st.info(f"💡 Found {len(isomers)} possible configurations.")

                st.subheader("2. 2D Structure Grid")
                labels = []
                for i, iso in enumerate(isomers):
                    Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                    stereo_info = []
                    
                    # E/Z
                    for bond in iso.GetBonds():
                        stereo = bond.GetStereo()
                        if stereo == Chem.BondStereo.STEREOE: stereo_info.append("E")
                        elif stereo == Chem.BondStereo.STEREOZ: stereo_info.append("Z")
                    
                    # R/S
                    centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                    for c in centers: stereo_info.append(f"{c[1]}")
                    
                    # Ra/Sa
                    allene_stereo = get_allene_stereo(iso)
                    if allene_stereo: stereo_info.extend(allene_stereo)
                    
                    label = f"Isomer {i+1}: " + (", ".join(stereo_info) if stereo_info else "Achiral")
                    labels.append(label)

                img = Draw.MolsToGridImage(isomers, molsPerRow=3, subImgSize=(300, 300), legends=labels)
                st.image(img, use_container_width=True)

                st.subheader("3. Interactive 3D Visualization")
                cols = st.columns(3)
                for i, iso in enumerate(isomers):
                    with cols[i % 3]:
                        render_3d(iso, labels[i])

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Advanced Mode: Axial Chirality Detection (Ra/Sa) for Allenes Fully Active.")
