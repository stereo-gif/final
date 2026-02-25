import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة والـ Sidebar
st.set_page_config(page_title="StereoMaster Pro 2026", layout="wide")

with st.sidebar:
    st.markdown("""
    <div style="background-color: #fdf2f2; padding: 15px; border-radius: 10px; border: 1px solid #800000;">
        <h3 style="color: #800000; font-family: serif;">Scientific Notes</h3>
        <p><b>1. Cis / Trans:</b> Relative side.</p>
        <p><b>2. E / Z:</b> Absolute (CIP System).</p>
        <p><b>3. R / S:</b> Chiral Centers.</p>
        <p><b>4. Ra / Sa:</b> Axial (Allenes).</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<h2 style='color: #800000; font-family: serif; border-bottom: 2px solid #dcdde1;'>Chemical Isomer Analysis System 2.0</h2>", unsafe_allow_html=True)

# --- دالة حساب Ra/Sa للألين ---
def get_allene_stereo(mol):
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

# --- دالة الرسم (الحل النهائي المضمون للـ Wedges) ---
def render_perfect_2d(mol):
    # إضافة الهيدروجين ضروري جداً عشان الـ Wedges تظهر زي الصورة
    m = Chem.AddHs(mol)
    AllChem.Compute2DCoords(m)
    # إجبار الـ RDKit على حساب الروابط الفراغية للرسم
    Chem.WedgeMolBonds(m, m.GetConformer())
    
    # الرسم باستخدام PIL Image (أكثر استقراراً في Streamlit)
    img = Draw.MolToImage(m, 
                          size=(400, 400), 
                          wedgeBonds=True, 
                          addStereoAnnotation=True,
                          legend="")
    return img

# --- منطق البرنامج الرئيسي ---
name = st.text_input("Enter Molecule Name:", "2,3-pentadiene")

if st.button("Generate Isomers"):
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            base_mol = Chem.MolFromSmiles(results[0].smiles)
            
            # كشف الألين وإجبار الكايراليتي
            pattern = Chem.MolFromSmarts("C=C=C")
            if base_mol.HasSubstructMatch(pattern):
                for match in base_mol.GetSubstructMatches(pattern):
                    base_mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

            opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers(base_mol, options=opts))
            
            # ضمان وجود أيزومرين للألين
            if len(isomers) == 1 and base_mol.HasSubstructMatch(pattern):
                iso2 = Chem.Mol(isomers[0])
                for a in iso2.GetAtoms():
                    tag = a.GetChiralTag()
                    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                isomers.append(iso2)

            # العرض في أعمدة
            cols = st.columns(len(isomers))
            for i, iso in enumerate(isomers):
                with cols[i]:
                    Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                    axial = get_allene_stereo(iso)
                    st.markdown(f"### Isomer {i+1}: <span style='color: #b22222;'>{axial}</span>", unsafe_allow_html=True)
                    
                    # عرض الـ 2D بالـ Wedges
                    st.image(render_perfect_2d(iso), use_container_width=True)
                    
                    # عرض الـ 3D
                    m3d = Chem.AddHs(iso)
                    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                    mblock = Chem.MolToMolBlock(m3d)
                    view = py3Dmol.view(width=350, height=300)
                    view.addModel(mblock, 'mol')
                    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                    view.zoomTo()
                    showmol(view)
        else:
            st.error("Compound not found.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
