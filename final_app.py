import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الواجهة
st.set_page_config(page_title="StereoMaster Pro 2026", layout="wide")

with st.sidebar:
    st.markdown("""
    <div style="background-color: #fdf2f2; padding: 15px; border-radius: 10px; border: 1px solid #800000;">
        <h3 style="color: #800000; font-family: serif;">Scientific Notes</h3>
        <p><b>1. Cis / Trans:</b> Relative side.</p>
        <p><b>2. E / Z:</b> Absolute priority.</p>
        <p><b>3. R / S:</b> Chiral Centers.</p>
        <p><b>4. Ra / Sa:</b> Axial (Allenes).</p>
    </div>
    """, unsafe_allow_html=True)

# دالة الرسم "الواضحة جداً" بدون أخطاء Attributes
def render_ultra_clear_2d(mol):
    m = Chem.AddHs(mol)
    AllChem.Compute2DCoords(m)
    Chem.WedgeMolBonds(m, m.GetConformer())
    
    # خيارات الرسم لزيادة الوضوح والسمك (بدون استخدام سمات غير معروفة)
    d_opts = Draw.MolDrawOptions()
    d_opts.addStereoAnnotation = True
    d_opts.bondLineWidth = 3.5       # زيادة سمك كل الروابط لجعلها Bold
    d_opts.minFontSize = 20          # تكبير حجم خط الذرات
    d_opts.annotationFontScale = 1.0 # حجم علامات R/S
    d_opts.prepareMolsBeforeDrawing = True 
    
    # استخدام MolToImage اللي بيتعامل مع الـ Wedges بشكل افتراضي عريض
    img = Draw.MolToImage(m, size=(500, 500), options=d_opts)
    return img

# دالة حساب Ra/Sa للألين
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

st.markdown("<h2 style='color: #800000;'>High-Contrast Isomer Analyzer</h2>", unsafe_allow_html=True)
name = st.text_input("Enter Molecule Name:", "1,3-dichloropropadiene")

if st.button("Generate Isomers"):
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            mol = Chem.MolFromSmiles(results[0].smiles)
            
            pattern = Chem.MolFromSmarts("C=C=C")
            if mol.HasSubstructMatch(pattern):
                for match in mol.GetSubstructMatches(pattern):
                    mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

            opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers(mol, options=opts))
            
            if len(isomers) == 1 and mol.HasSubstructMatch(pattern):
                iso2 = Chem.Mol(isomers[0])
                for a in iso2.GetAtoms():
                    tag = a.GetChiralTag()
                    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                isomers.append(iso2)

            cols = st.columns(len(isomers))
            for i, iso in enumerate(isomers):
                with cols[i]:
                    Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                    axial = get_allene_stereo(iso)
                    st.markdown(f"### Isomer {i+1}: <span style='color: #800000;'>{axial}</span>", unsafe_allow_html=True)
                    
                    # الرسم بالسمك الجديد
                    st.image(render_ultra_clear_2d(iso), use_container_width=True)
                    
                    # الـ 3D للتأكيد
                    m3d = Chem.AddHs(iso)
                    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                    mblock = Chem.MolToMolBlock(m3d)
                    view = py3Dmol.view(width=300, height=300)
                    view.addModel(mblock, 'mol')
                    view.setStyle({'stick': {'width': 6}, 'sphere': {'scale': 0.3}})
                    view.zoomTo()
                    showmol(view)
        else:
            st.error("Compound not found.")
    except Exception as e:
        st.error(f"Error details: {e}")
