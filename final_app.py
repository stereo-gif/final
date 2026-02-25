import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Professional Isomer Analyzer", layout="wide")

# 2. عرض النوت (Reference Guide) في واجهة التطبيق الرئيسية
st.markdown("""
<div style="background-color: #fdf2f2; padding: 20px; border-radius: 15px; border-left: 8px solid #800000; margin-bottom: 25px;">
    <h3 style="color: #800000; font-family: serif; margin-top: 0;">🧪 Stereoisomerism Reference Guide</h3>
    <table style="width:100%; border:none; color: #333;">
        <tr>
            <td><b>1. Cis / Trans:</b> Relative side arrangement.</td>
            <td><b>2. E / Z (CIP):</b> Absolute priority system.</td>
        </tr>
        <tr>
            <td><b>3. R / S:</b> Optical chirality centers.</td>
            <td><b>4. Ra / Sa (Axial):</b> Allenes (C=C=C systems).</td>
        </tr>
    </table>
    <p style="margin-bottom: 0; margin-top: 10px; font-size: 0.9em; color: #666;">
        <i>*Note: E/Z is required when all 4 groups on the double bond are different.</i>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<h2 style='color: #800000; font-family: serif;'>Professional Isomer Analysis System</h2>", unsafe_allow_html=True)

# دالة الرسم الذكية (ألين واضح وباقي المركبات رقيقة)
def render_smart_2d(mol):
    is_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("C=C=C"))
    m = Chem.AddHs(mol) if is_allene else Chem.RemoveHs(mol)
    
    if AllChem.EmbedMolecule(m, AllChem.ETKDG()) != -1:
        AllChem.Compute2DCoords(m)
        Chem.WedgeMolBonds(m, m.GetConformer())
    else:
        AllChem.Compute2DCoords(m)

    d_opts = Draw.MolDrawOptions()
    d_opts.addStereoAnnotation = True
    
    if is_allene:
        d_opts.bondLineWidth = 3.0
        d_opts.minFontSize = 18
    else:
        d_opts.bondLineWidth = 1.6
        d_opts.minFontSize = 14

    img = Draw.MolToImage(m, size=(500, 500), options=d_opts)
    return img

# دالة حساب Ra/Sa
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

# المدخلات
name = st.text_input("Enter Molecule Name (e.g., 2,3-pentadiene or Tartaric acid):", "2,3-pentadiene")

if st.button("Analyze Structure"):
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            base_mol = Chem.MolFromSmiles(results[0].smiles)
            pattern = Chem.MolFromSmarts("C=C=C")
            
            if base_mol.HasSubstructMatch(pattern):
                for match in base_mol.GetSubstructMatches(pattern):
                    base_mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

            opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers(base_mol, options=opts))
            
            if len(isomers) == 1 and base_mol.HasSubstructMatch(pattern):
                iso2 = Chem.Mol(isomers[0])
                for a in iso2.GetAtoms():
                    tag = a.GetChiralTag()
                    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                isomers.append(iso2)

            st.write(f"Showing {len(isomers)} possible stereoisomers:")
            
            cols = st.columns(len(isomers))
            for i, iso in enumerate(isomers):
                with cols[i]:
                    Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                    axial = get_allene_stereo(iso)
                    st.markdown(f"#### Isomer {i+1}: <span style='color: #800000;'>{axial}</span>", unsafe_allow_html=True)
                    
                    # عرض الرسم المطور
                    st.image(render_smart_2d(iso), use_container_width=True)
                    
                    # عرض الـ 3D
                    m3d = Chem.AddHs(iso)
                    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                    mblock = Chem.MolToMolBlock(m3d)
                    view = py3Dmol.view(width=300, height=300)
                    view.addModel(mblock, 'mol')
                    view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                    view.zoomTo()
                    showmol(view)
        else:
            st.error("Compound not found in database.")
    except Exception as e:
        st.error(f"Analysis Error: {e}")
