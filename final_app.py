import streamlit as st
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from stmol import showmol
import py3Dmol
import numpy as np

# 1. إعدادات الصفحة والـ Sidebar (مرجع علمي)
st.set_page_config(page_title="StereoMaster 2.0", layout="wide")

with st.sidebar:
    st.markdown("""
    <div style="background-color: #fdf2f2; padding: 15px; border-radius: 10px; border: 1px solid #800000;">
        <h3 style="color: #800000; font-family: serif;">Isomerism Notes</h3>
        <p><b>1. Cis / Trans:</b> Relative side.</p>
        <p><b>2. E / Z:</b> Absolute (CIP).</p>
        <p><b>3. R / S:</b> Chiral Centers.</p>
        <p><b>4. Ra / Sa:</b> Axial (Allenes).</p>
    </div>
    """, unsafe_allow_html=True)

# --- دالة الرسم الاحترافية (Wedge/Dash Force) ---
def draw_mol_with_wedges(mol, size=(400, 400)):
    # 1. تجهيز الجزيء وإضافة الهيدروجين (أساسي عشان تظهر زي الصورة)
    m = Chem.AddHs(mol)
    AllChem.EmbedMolecule(m, AllChem.ETKDG()) # توليد شكل فراغي
    AllChem.Compute2DCoords(m) # تحويله لـ 2D مع الحفاظ على الـ Stereo
    
    # 2. تحديد نوع الروابط (Wedge/Dash) برمجياً بناءً على الـ 3D
    Chem.WedgeMolBonds(m, m.GetConformer())
    
    # 3. استخدام الرسام المتقدم SVG
    drawer = rdMolDraw2D.MolDraw2DSvg(size[0], size[1])
    options = drawer.drawOptions()
    options.addStereoAnnotation = True
    options.atomLabelFontSize = 20
    options.bondLineWidth = 3
    
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, m)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

# --- دالة حساب Ra/Sa ---
def get_allene_stereo(mol):
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

# --- واجهة التطبيق الرئيسية ---
st.markdown("<h2 style='color: #800000;'>Advanced Isomer Analyzer</h2>", unsafe_allow_html=True)
name = st.text_input("Enter Molecule Name:", "2,3-pentadiene")

if st.button("Generate Isomers"):
    results = pcp.get_compounds(name, 'name')
    if results:
        mol = Chem.MolFromSmiles(results[0].smiles)
        
        # إجبار الألين على إظهار الأيزومرات
        pattern = Chem.MolFromSmarts("C=C=C")
        if mol.HasSubstructMatch(pattern):
            for match in mol.GetSubstructMatches(pattern):
                mol.GetAtomWithIdx(match[0]).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

        opts = StereoEnumerationOptions(tryEmbedding=True, onlyUnassigned=False)
        isomers = list(EnumerateStereoisomers(mol, options=opts))
        
        # إضافة النسخة المرآة يدوياً لضمان Ra و Sa
        if len(isomers) == 1:
            iso2 = Chem.Mol(isomers[0])
            for a in iso2.GetAtoms():
                if a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
                elif a.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW: a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            isomers.append(iso2)

        cols = st.columns(len(isomers))
        for i, iso in enumerate(isomers):
            with cols[i]:
                axial = get_allene_stereo(iso)
                st.markdown(f"### Isomer {i+1}: **{axial}**")
                
                # عرض الـ 2D بالـ Wedges (زي الصورة)
                svg = draw_mol_with_wedges(iso)
                st.write(svg, unsafe_allow_html=True)
                
                # عرض الـ 3D للتأكيد
                m3d = Chem.AddHs(iso)
                AllChem.EmbedMolecule(m3d, AllChem.ETKDG())
                mblock = Chem.MolToMolBlock(m3d)
                view = py3Dmol.view(width=300, height=300)
                view.addModel(mblock, 'mol')
                view.setStyle({'stick': {}, 'sphere': {'scale': 0.3}})
                view.zoomTo()
                showmol(view, height=300, width=300)
