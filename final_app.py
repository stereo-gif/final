import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة
st.set_page_config(page_title="Chemical Isomer Analysis Pro", layout="wide")

# 2. تصميم الواجهة والـ Sidebar (مرجع علمي ثابت)
with st.sidebar:
    st.markdown(f"""
    <div style="background-color: #fdf2f2; padding: 15px; border-radius: 10px; border: 1px solid #800000;">
        <h3 style="color: #800000; font-family: serif;">Scientific Notes</h3>
        <p><b>1. Cis / Trans:</b> Relative side.</p>
        <p><b>2. E / Z (CIP):</b> Absolute priority.</p>
        <p><b>3. R / S (Optical):</b> Chiral centers.</p>
        <p><b>4. Ra / Sa (Axial):</b> Allenes (C=C=C).</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("💡 Note: E/Z is required when all 4 groups on the double bond are different.")

st.markdown("""
<style>
    .stApp { background-color: white; color: black; }
</style>
<h2 style='color: #800000; font-family: serif; border-bottom: 2px solid #dcdde1;'>Chemical Isomer Analysis System 2.0</h2>
""", unsafe_allow_html=True)

# دالة حساب Ra/Sa للألين
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

# دالة عرض الـ 3D
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

# 3. مدخلات المستخدم
compound_name = st.text_input("Enter Structure Name:", "2,3-pentadiene")

if st.button("Analyze & Visualize"):
    try:
        results = pcp.get_compounds(compound_name, 'name')
        if results:
            smiles = results[0].smiles
            mol = Chem.MolFromSmiles(smiles)
            
            # إجبار الألين على الكايراليتي
            pattern = Chem.MolFromSmarts("C=C=C")
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            
            opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
            isomers = list(EnumerateStereoisomers(mol, options=opts))
            
            # خلق الأيزومر المرآة يدوياً للألين لو مظهرش
            if len(isomers) == 1:
                iso2 = Chem.Mol(isomers[0])
                for atom in iso2.GetAtoms():
                    tag = atom.GetChiralTag()
                    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                    elif tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                isomers.append(iso2)

            st.subheader(f"Total Isomers Found: {len(isomers)}")
            
            labels = []
            for i, iso in enumerate(isomers):
                # أهم خطوة: حساب الاستيريو كيمستري وتجهيز الروابط (Wedges/Dashes)
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                
                # كشف الأيزومرات
                axial = get_allene_stereo(iso)
                centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                
                stereo_text = []
                if centers: stereo_text.extend([f"{c[1]}" for c in centers])
                if axial: stereo_text.extend(axial)
                
                label = f"Isomer {i+1}: {', '.join(stereo_text) if stereo_text else 'Achiral'}"
                labels.append(label)

            # --- رسم الـ 2D Grid مع الـ Hatched Bonds ---
            # استخدام دالة DrawOptions لتوضيح الروابط الفراغية
            d_opts = Draw.MolDrawOptions()
            d_opts.addStereoAnnotation = True # كتابة R/S على الذرات
            
            img = Draw.MolsToGridImage(isomers, 
                                       molsPerRow=2, 
                                       subImgSize=(400, 400), 
                                       legends=labels,
                                       useSVG=False, # PNG بيدعم الـ Wedge بشكل مستقر
                                       drawOptions=d_opts)
            st.image(img, use_container_width=True)

            # عرض الـ 3D
            st.divider()
            cols = st.columns(len(isomers))
            for i, iso in enumerate(isomers):
                with cols[i]:
                    render_3d(iso, labels[i])
                    
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.caption("Advanced 2D/3D Stereochemistry Visualization Engine Active.")
