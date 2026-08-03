import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageDraw
import pytesseract
import re
import difflib

# ==========================================
# 1. ตั้งค่าหน้าเพจ 
# ==========================================
st.set_page_config(page_title="PetFood Safety AI", page_icon="🐶", layout="wide") 

# ==========================================
# 2. ตั้งค่าธีม (Pet-Friendly Theme)
# ==========================================
def set_theme():
    css = """
    <style>
    /* พื้นหลังสีสว่างสบายตา */
    .stApp {
        background-color: #f4f6f9; 
    }
    
    /* กล่องเนื้อหาหลัก */
    .main .block-container {
        background-color: #FFFFFF !important;
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border-top: 8px solid #ff9f43; /* ขีดสีส้มด้านบนให้ดูเป็นแอปสัตว์เลี้ยง */
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* เปลี่ยนสีปุ่มเป็นสีส้ม */
    .stButton>button {
        border-radius: 20px;
        font-weight: bold;
        background-color: #ff9f43;
        color: white !important;
        border: none;
    }
    .stButton>button:hover {
        background-color: #e67e22;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_theme()

# ==========================================
# 3. โหลดฐานข้อมูล
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('database.csv')
        df['ingredient'] = df['ingredient'].astype(str).str.lower().str.strip()
        return df
    except FileNotFoundError:
        st.error("❌ ไม่พบไฟล์ 'database.csv'")
        return pd.DataFrame()

df_db = load_data()

# ==========================================
# 4. ฟังก์ชันวิเคราะห์ส่วนผสม (ระดับ Pro - Spatial Mapping + Regex)
# ==========================================
def analyze_ingredients_with_boxes(processed_img, df):
    data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DATAFRAME)
    # กรองเอาเฉพาะแถวที่มีข้อความ และ Reset Index เพื่อให้ใช้อ้างอิงตำแหน่งได้แม่นยำ
    data = data[data.text.notnull() & (data.text.str.strip() != "")].reset_index(drop=True)
    
    if data.empty:
        return pd.DataFrame(), data
        
    # -------------------------------------------------------------
    # เทคนิคที่ 1: Character-to-Box Mapping (เชื่อมตัวอักษรเข้ากับพิกัด)
    # -------------------------------------------------------------
    full_text = ""
    char_to_word_idx = [] # เก็บว่าตัวอักษรที่ i มาจาก DataFrame แถวไหน
    
    for idx, row in data.iterrows():
        word = str(row['text']).lower().strip()
        # นำคำมาต่อกันด้วยช่องว่าง
        full_text += word + " "
        # จับคู่ตัวอักษรทุกตัว (รวมถึงช่องว่าง) ให้ชี้กลับไปที่พิกัด Index ของคำนั้น
        char_to_word_idx.extend([idx] * (len(word) + 1))
        
    found_ingredients = []
    
    # ดิกชันนารีคำพ้องความหมาย (ทำเป็น List เพื่อให้รองรับชื่อสารได้หลายรูปแบบ)
    reverse_synonyms = {
        "tocopherol": ["vitamin e", "mixed-tocopherols", "mixed tocopherols"],
        "ascorbic acid": ["vitamin c", "l-ascorbyl-2-polyphosphate"],
        "meat by-product": ["meat by-products", "chicken by-product", "chicken by-products", "poultry by-product"],
        "artificial color": ["added color", "color added", "yellow 5", "red 40", "yellow 6", "blue 2", "yellow 6, blue 2"] 
    }
    
    for index, db_row in df.iterrows():
        ing_name = str(db_row['ingredient']).strip().lower()
        
        search_terms = [ing_name]
        if ing_name in reverse_synonyms:
            search_terms.extend(reverse_synonyms[ing_name])
            
        for term in search_terms:
            # -------------------------------------------------------------
            # เทคนิคที่ 2: Flexible Regex Matching (ค้นหาแบบยืดหยุ่น)
            # -------------------------------------------------------------
            # แปลงคำค้นหาให้ยืดหยุ่น เช่น "chicken meal" จะเจอทั้ง "chicken meal", "chicken-meal", "chicken, meal"
            escaped_words = [re.escape(w) for w in term.split()]
            pattern = r'[\s\W]*'.join(escaped_words) 
            
            # ค้นหาคำจากข้อความยาวที่ต่อกันไว้แล้ว
            match = re.search(pattern, full_text)
            
            if match:
                # ถ้าเจอ ให้ดึงตำแหน่งตัวอักษรเริ่มต้นและสิ้นสุดออกมา
                start_char_idx = match.start()
                end_char_idx = match.end() - 1 
                
                # ป้องกัน Index ทะลุ
                end_char_idx = min(end_char_idx, len(char_to_word_idx) - 1)
                
                # โยงตำแหน่งตัวอักษร กลับไปหาพิกัดกรอบ (Bounding Box) ใน DataFrame
                start_word_idx = char_to_word_idx[start_char_idx]
                end_word_idx = char_to_word_idx[end_char_idx]
                
                # ดึงแถวข้อมูลทั้งหมดที่ครอบคลุมคำนี้
                matched_rows = data.loc[start_word_idx:end_word_idx]
                
                if not matched_rows.empty:
                    # คำนวณกรอบที่ครอบคลุมข้อความทั้งหมด (แม้จะถูกตัดขึ้นบรรทัดใหม่)
                    min_x = matched_rows['left'].min()
                    min_y = matched_rows['top'].min()
                    max_x = (matched_rows['left'] + matched_rows['width']).max()
                    max_y = (matched_rows['top'] + matched_rows['height']).max()
                    
                    best_box = (min_x, min_y, max_x - min_x, max_y - min_y)
                    
                    existing_ings = [x['Ingredient'].lower() for x in found_ingredients]
                    if ing_name not in existing_ings:
                        found_ingredients.append({
                            'Ingredient': ing_name.title(),
                            'Function': db_row['function'],
                            'Risk': db_row['risk_level'],
                            'box': best_box
                        })
                # ถ้าเจอแล้ว ให้ข้ามไปหาสารตัวต่อไปเลย ไม่ต้องค้นหาคำพ้องซ้ำ
                break 

    if found_ingredients:
        result_df = pd.DataFrame(found_ingredients)
        result_df = result_df.drop_duplicates(subset=['Ingredient']).reset_index(drop=True)
        return result_df, data
    else:
        return pd.DataFrame(), data


# ==========================================
# 5. หน้าจอหลัก (UI)
# ==========================================
col_icon, col_title = st.columns([1, 9])
with col_icon:
    st.image("woman_5362023.png", width=65) # สามารถเปลี่ยนชื่อไฟล์ตรงนี้เป็นรูปหมา/แมวได้
with col_title:
    st.title("PetFood Safety AI")

st.markdown("**ระบบสแกนและตรวจจับส่วนผสมอันตรายในอาหารสัตว์เลี้ยง (พร้อมระบบคลิกไฮไลต์ตำแหน่งบนถุงอาหาร)**")
st.markdown("---")

# 🟢 ฟีเจอร์ "ว้าว": ให้ผู้ใช้เลือกสิ่งที่สัตว์เลี้ยงแพ้
st.markdown("### ⚠️ ตั้งค่าสุขภาพสัตว์เลี้ยง (AI Personalization)")
user_allergies = st.multiselect(
    "สัตว์เลี้ยงของคุณมีประวัติแพ้วัตถุดิบอะไรบ้าง? (ระบบจะแจ้งเตือนหากสแกนพบ)",
    ["Chicken", "Beef", "Corn", "Wheat", "Soy", "Fish Meal", "Dairy"]
)
st.markdown("---")

tab1, tab2 = st.tabs(["📸 ถ่ายรูปจากกล้อง", "📂 อัปโหลดรูปภาพ"])

with tab1:
    st.info("💡 **วิธีใช้งาน:** ถ่ายรูปฉลากส่วนผสมให้ชัดเจน แล้วรอ AI ประมวลผล")
    camera_file = st.camera_input("ถ่ายรูปฉลากผลิตภัณฑ์")
    
with tab2:
    st.info("💡 **วิธีใช้งาน:** อัปโหลดรูปภาพฉลากผลิตภัณฑ์ แล้วรอ AI ประมวลผล")
    uploaded_file = st.file_uploader("เลือกรูปภาพ...", type=['jpg', 'jpeg', 'png'])

img_file = camera_file if camera_file is not None else uploaded_file

if img_file is not None:
    original_image = Image.open(img_file)
    
    with st.spinner('🤖 AI กำลังอ่านข้อความและประมวลผลตำแหน่งพิกัด...'):
        try:
            gray_img = original_image.convert('L')
            enhancer_contrast = ImageEnhance.Contrast(gray_img)
            processed_img = enhancer_contrast.enhance(1.5)
            
            result_df, ocr_data = analyze_ingredients_with_boxes(processed_img, df_db)
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
            result_df = pd.DataFrame()

    if result_df.empty:
        st.warning("สแกนพบภาพ แต่ไม่พบสารสำคัญที่ตรงกับฐานข้อมูล แนะนำให้ถ่ายรูปในมุมที่สว่างและชัดเจนขึ้น")
    else:
        st.success(f"✅ ตรวจพบวัตถุดิบที่รู้จัก {len(result_df)} ชนิด")

        # 🟢 เพิ่มระบบประมวลผลการแพ้อาหาร (Personalized Alert) ตรงนี้
        allergy_alerts = []
        for index, row in result_df.iterrows():
            if row['Ingredient'] in user_allergies:
                allergy_alerts.append(row['Ingredient'])
        
        if allergy_alerts:
            st.error(f"🚨 **แจ้งเตือนอันตรายรุนแรง!** ตรวจพบส่วนผสมที่สัตว์เลี้ยงของคุณแพ้: **{', '.join(allergy_alerts)}**")
        elif len(user_allergies) > 0:
            st.success("✨ **ปลอดภัย!** ไม่พบส่วนผสมที่สัตว์เลี้ยงของคุณแพ้ในอาหารถุงนี้")

        st.markdown("### 🔍 คลิกเลือกสารเพื่อดูตำแหน่งไฮไลต์บนรูปภาพ")
        
        selected_ingredient = st.selectbox(
            "เลือกสารที่ต้องการตรวจสอบตำแหน่ง:",
            options=result_df['Ingredient'].tolist()
        )

        col_img, col_res = st.columns([1, 1])

        with col_img:
            draw_image = original_image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            target_row = result_df[result_df['Ingredient'] == selected_ingredient].iloc[0]
            bx, by, bw, bh = target_row['box']
            
            pad = 5
            draw.rectangle(
                [bx - pad, by - pad, bx + bw + pad, by + bh + pad],
                outline="red",
                width=4
            )
            
            st.image(draw_image, caption=f"ตำแหน่งของสาร: {selected_ingredient}", use_container_width=True)
            st.caption("🔴 กรอบสีแดงบนรูปภาพคือตำแหน่งที่ AI ตรวจพบวัตถุดิบตัวนี้ครับ")

        with col_res:
            st.markdown("### 📋 ผลการวิเคราะห์แยกตามระดับความเสี่ยง")
            
            safe_df = result_df[result_df['Risk'] == 'Safe']
            warn_df = result_df[result_df['Risk'] == 'Warning']
            danger_df = result_df[result_df['Risk'] == 'Danger']
            
            with st.expander(f"🟢 ปลอดภัย ({len(safe_df)} ชนิด)", expanded=True):
                for _, row in safe_df.iterrows():
                    highlight_mark = " 👉 (กำลังแสดงตำแหน่ง)" if row['Ingredient'] == selected_ingredient else ""
                    st.write(f"- **{row['Ingredient']}**{highlight_mark}<br><small>{row['Function']}</small>", unsafe_allow_html=True)
                    
            with st.expander(f"🟡 เฝ้าระวัง ({len(warn_df)} ชนิด)", expanded=True):
                for _, row in warn_df.iterrows():
                    highlight_mark = " 👉 (กำลังแสดงตำแหน่ง)" if row['Ingredient'] == selected_ingredient else ""
                    st.write(f"- **{row['Ingredient']}**{highlight_mark}<br><small>{row['Function']}</small>", unsafe_allow_html=True)
                    
            with st.expander(f"🔴 อันตราย ({len(danger_df)} ชนิด)", expanded=True):
                for _, row in danger_df.iterrows():
                    highlight_mark = " 👉 (กำลังแสดงตำแหน่ง)" if row['Ingredient'] == selected_ingredient else ""
                    st.write(f"- **{row['Ingredient']}**{highlight_mark}<br><small>{row['Function']}</small>", unsafe_allow_html=True)

st.markdown("---")
st.caption("📝 **หมายเหตุ:** ข้อมูลนี้เป็นเพียงการวิเคราะห์เบื้องต้น หากสัตว์เลี้ยงมีอาการผิดปกติหลังทานอาหาร ควรปรึกษาสัตวแพทย์ทันที")
