import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة والتصميم (نفس اللي بتحبيه)
st.set_page_config(page_title="Advanced Chemical Isomer Analysis", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: white; color: black; }
</style>
<h2 style='color: #800000; font-family: serif; border-bottom: 2px solid #dcdde1;'>Chemical Isomer Analysis System 2.0</h2>
<div style="background-color: #f9f9f9; padding: 15px; border: 1px solid #e1e1e1; border-left: 4px solid #800000; margin-bottom: 20px;">
    <strong style="color: #800000;">Stereoisomerism Reference Guide:</strong><br>
    1. <b>Cis / Trans</b> | 2. <b>E / Z</b> | 3. <b>R / S</b> | 4. <b>Ra / Sa (Allenes)</b>
</div>
""", unsafe_allow_html=True)

# دالة حساب Ra/Sa (محدثة لضمان الدقة)
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
                    if l_subs and r_subs:
                        p1, p3 = np.array(conf.GetAtomPosition(a1.GetIdx())), np.array(conf.GetAtomPosition(a3.GetIdx()))
                        pl, pr = np.array(conf.GetAtomPosition(l_subs[0].GetIdx())), np.array(conf.GetAtomPosition(r_subs[0].GetIdx()))
                        dot = np.dot(np.cross(pl-p1, p3-p1), pr-p3)
                        results.append("Ra" if dot > 0 else "Sa")
    return results

def render_3d(mol, title):
    mol_3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
    mblock = Chem.MolToMolBlock(mol_3d)
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mblock, 'mol')
    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    st.write(f"**{title}**")
    showmol(view, height=300, width=400)

compound_name = st.text_input("Enter Structure Name:", "2,3-pentadiene")

if st.button("Analyze & Visualize"):
    try:
        results = pcp.get_compounds(compound_name, 'name')
        if results:
            smiles = results[0].smiles
            mol = Chem.MolFromSmiles(smiles)
            
            # --- الحيلة البرمجية لإجبار الألين على إظهار أيزومراته ---
            # نقوم بالبحث عن نظام C=C=C وتحديد ذراته كأهداف للـ Stereo
            pattern = Chem.MolFromSmarts("C=C=C")
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                # ذرات الأطراف في الألين (0 و 2) نعطيها علامة chiral تجريبية
                mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            # الآن نولد الأيزومرات
            opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers(mol, options=opts))
            
            # لو طلع لنا أيزومر واحد بس (بسبب الـ SMILES الأصلي)، هنخلق التاني يدوياً بالـ Mirror
            if len(isomers) == 1:
                iso2 = Chem.Mol(isomers[0])
                # عكس كل مراكز الاستيريو يدوياً
                for atom in iso2.GetAtoms():
                    if atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
                        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    elif atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
                        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                isomers.append(iso2)

            st.subheader(f"Found {len(isomers)} Stereoisomers")
            
            labels = []
            for i, iso in enumerate(isomers):
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                axial = get_allene_stereo(iso)
                label = f"Isomer {i+1}: {', '.join(axial) if axial else 'Achiral'}"
                labels.append(label)

            # عرض الـ Grid
            img = Draw.MolsToGridImage(isomers, molsPerRow=2, subImgSize=(400, 400), legends=labels)
            st.image(img, use_container_width=True)

            # عرض الـ 3D
            cols = st.columns(len(isomers))
            for i, iso in enumerate(isomers):
                with cols[i]:
                    render_3d(iso, labels[i])
                    
    except Exception as e:
        st.error(f"Error: {e}")
