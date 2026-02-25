import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, EnumerateStereoisomers
from stmol import showmol
import py3Dmol
import numpy as np

# ==============================
# 1. دالة الـ Ra/Sa (للالين)
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
# 2. دالة الـ 3D 
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
# 3. واجهة البرنامج (Streamlit)
# ==============================
st.set_page_config(layout="wide")
st.title("Full Stereoisomer Gallery (2D & 3D)")

# النوت العلمية المحفوظة
with st.sidebar:
    st.header("Saved Reference Guide")
    st.markdown("""
    - **Cis/Trans**: Relative side.
    - **E/Z**: Absolute (CIP Rules).
    - **R/S**: Chiral center.
    - **Ra/Sa**: Axial (Allenes).
    """)

name = st.text_input("Enter Molecule Name:", "Thalidomide")

if st.button("Generate Gallery"):
    try:
        results = pcp.get_compounds(name, 'name')
        if results:
            base_mol = Chem.MolFromSmiles(results[0].smiles)
            
            # توليد كل الأيزومرات الممكنة
            opts = EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=True)
            isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(base_mol, options=opts))
            
            st.success(f"Found {len(isomers)} potential isomers.")
            
            # --- المرحلة الأولى: عرض كل الـ Isomers في Grid واحدة ---
            st.subheader("2D Comparison Grid")
            grid_labels = []
            processed_isomers = []
            
            for i, iso in enumerate(isomers):
                Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                axial = get_allene_label(iso)
                
                label = f"Isomer {i+1}\nRS:{centers}"
                if axial: label += f"\nAxial:{axial}"
                
                grid_labels.append(label)
                processed_isomers.append(iso)
            
            # رسم الشبكة (Grid)
            img = Draw.MolsToGridImage(processed_isomers, 
                                       molsPerRow=3, 
                                       subImgSize=(300, 300), 
                                       legends=grid_labels,
                                       useSVG=False) # PNG أسرع في الـ Grid
            st.image(img, use_container_width=True)
            
            st.divider()
            
            # --- المرحلة الثانية: عرض التفاصيل والـ 3D لكل واحد ---
            st.subheader("Detailed 3D Analysis")
            for i, iso in enumerate(processed_isomers):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"### Isomer {i+1}")
                    st.write(f"**Configuration:** {grid_labels[i]}")
                    # إعادة رسم 2D مكبرة
                    st.image(Draw.MolToImage(iso, size=(400, 400), wedgeBonds=True))
                with col2:
                    render_3d(iso)
                st.divider()
                
        else:
            st.error("Compound not found.")
    except Exception as e:
        st.error(f"Error: {e}")
