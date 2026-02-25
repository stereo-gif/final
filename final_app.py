import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, EnumerateStereoisomers
from rdkit.Chem.Draw import rdMolDraw2D
from stmol import showmol
import py3Dmol

# ==============================
# 1. تهيئة الصفحة والنوت العلمية
# ==============================
st.set_page_config(page_title="StereoMaster Pro 2026", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Chemical Isomer Expert</h1>", unsafe_allow_html=True)

with st.expander("📚 Stereoisomerism Quick Reference (Saved)"):
    st.markdown("""
    * **Cis / Trans**: Relative orientation of identical groups.
    * **E / Z**: Absolute orientation based on **CIP Priority** (Atomic Number).
        - **Z (Zusammen)**: High priority groups on the *Same* side.
        - **E (Entgegen)**: High priority groups on *Opposite* sides.
    * **R / S**: Absolute configuration of chiral centers (Clockwise/Counter-clockwise).
    """)

# ==============================
# 2. دالة الرسم الـ 2D المحسنة (High Quality SVG)
# ==============================
def render_pretty_2d(mol, label):
    # تحضير الجزيء وعمل الكايراليتي
    mc = Chem.Mol(mol)
    AllChem.Compute2DCoords(mc)
    
    # إعداد الرسام SVG
    drawer = rdMolDraw2D.MolDraw2DSvg(400, 400)
    options = drawer.drawOptions()
    options.addStereoAnnotation = True  # دي اللي بتكتب R/S و E/Z على الرسمة
    options.atomLabelFontSize = 25
    options.bondLineWidth = 3
    options.continuousHighlight = False
    
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    st.write(f"**{label}**")
    st.image(svg, use_container_width=True)

# ==============================
# 3. دالة العرض الـ 3D
# ==============================
def render_3d_structure(mol):
    m3d = Chem.AddHs(mol)
    AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(m3d) # تحسين الشكل طاقياً
    mblock = Chem.MolToMolBlock(m3d)
    
    view = py3Dmol.view(width=400, height=300)
    view.addModel(mblock, 'mol')
    view.setStyle({'stick': {'colorscheme': 'Jmol', 'radius': 0.2}, 'sphere': {'scale': 0.3}})
    view.zoomTo()
    showmol(view, height=300, width=400)

# ==============================
# 4. المنطق الأساسي للبرنامج
# ==============================
name = st.text_input("Enter Molecule Name:", "Thalidomide")

if st.button("Analyze Structure"):
    if name:
        with st.spinner("Searching PubChem and calculating isomers..."):
            try:
                # جلب البيانات
                compounds = pcp.get_compounds(name, 'name')
                if not compounds:
                    st.error("Could not find this molecule.")
                else:
                    smiles = compounds[0].smiles
                    base_mol = Chem.MolFromSmiles(smiles)
                    
                    # توليد الأيزومرات (R/S)
                    opts = EnumerateStereoisomers.StereoEnumerationOptions(tryEmbedding=True)
                    isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(base_mol, options=opts))
                    
                    st.success(f"Found {len(isomers)} potential stereoisomers.")
                    
                    # تقسيم الشاشة لأعمدة
                    for i, iso in enumerate(isomers):
                        # حساب الكايراليتي لكل أيزومر
                        Chem.AssignStereochemistry(iso, force=True, cleanIt=True)
                        centers = Chem.FindMolChiralCenters(iso, includeUnassigned=True)
                        
                        st.divider()
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            # عرض الـ 2D المحسن
                            render_pretty_2d(iso, f"Isomer {i+1}: {centers}")
                            
                        with col2:
                            # عرض الـ 3D التفاعلي
                            st.write("**3D Interactive Model**")
                            render_3d_structure(iso)
                            
            except Exception as e:
                st.error(f"Error processing molecule: {e}")
    else:
        st.warning("Please enter a name first.")
