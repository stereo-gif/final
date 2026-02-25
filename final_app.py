import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, EnumerateStereoisomers
from stmol import showmol
import py3Dmol
import numpy as np

# ==============================
# دالة الـ 3D اللي كانت شغالة معاكي (ملمستهاش)
# ==============================
def render_3d(mol):
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
    mblock = Chem.MolToMolBlock(mol_3d)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mblock, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    showmol(view, height=300, width=400)

# ==============================
# دالة الـ Allene (بقت بتشتغل في الخلفية فقط)
# ==============================
def get_axial_label(mol):
    try:
        m = Chem.AddHs(mol)
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
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
# الواجهة (Streamlit Interface)
# ==============================
st.set_page_config(layout="wide")
st.title("Isomer & Stereo Analyzer (Fixed)")

compound_name = st.text_input("Structure Name:", "2,3-pentadiene")

if st.button("Run Analysis"):
    results = pcp.get_compounds(compound_name, 'name')
    if results:
        base_mol = Chem.MolFromSmiles(results[0].smiles)
        
        # أهم تعديل للـ R/S:
        opts = EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=True)
        isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(base_mol, options=opts))
        
        st.write(f"Analyzed {len(isomers)} isomers:")
        
        cols = st.columns(2)
        for i, iso in enumerate(isomers):
            with cols[i % 2]:
                # تحديد الـ R/S والـ Allene
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                rs_centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                axial = get_axial_label(iso)
                
                # النص اللي هيظهر للمستخدم
                info_text = f"Isomer {i+1} | R/S: {rs_centers}"
                if axial: info_text += f" | Axial: {axial}"
                
                st.subheader(info_text)
                
                # عرض 2D
                st.image(Draw.MolToImage(iso, size=(400, 400), legend=info_text))
                
                # عرض 3D
                render_3d(iso)
                st.divider()
