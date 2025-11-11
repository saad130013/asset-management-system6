import pandas as pd
import numpy as np
import json
from datetime import datetime
import base64

def format_currency(amount, currency="ريال"):
    """تنسيق الأرقام كعملة"""
    if pd.isna(amount):
        return "غير متوفر"
    
    try:
        if amount >= 1e6:
            return f"{amount/1e6:.2f} مليون {currency}"
        elif amount >= 1e3:
            return f"{amount/1e3:.1f} ألف {currency}"
        else:
            return f"{amount:,.0f} {currency}"
    except:
        return str(amount)

def validate_data(df, required_columns=None):
    """التحقق من صحة البيانات"""
    validation_results = {
        'is_valid': True,
        'issues': [],
        'warnings': [],
        'summary': {}
    }
    
    if df is None or df.empty:
        validation_results['is_valid'] = False
        validation_results['issues'].append("البيانات فارغة أو غير محملة")
        return validation_results
    
    # التحقق من الأعمدة المطلوبة
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['issues'].append(f"الأعمدة المفقودة: {missing_columns}")
            validation_results['is_valid'] = False
    
    # التحقق من القيم الفارغة
    null_counts = df.isnull().sum()
    high_null_columns = null_counts[null_counts > len(df) * 0.5].index.tolist()
    
    if high_null_columns:
        validation_results['warnings'].append(f"أعمدة تحتوي على أكثر من 50% قيم فارغة: {high_null_columns}")
    
    # إحصائيات عامة
    validation_results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'total_null_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return validation_results

def export_to_excel(df, filename=None):
    """تصدير DataFrame إلى ملف Excel"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"تقرير_الأصول_{timestamp}.xlsx"
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # البيانات الأساسية
            df.to_excel(writer, sheet_name='البيانات الأساسية', index=False)
            
            # ملخص الإحصائيات
            summary_data = []
            
            # إحصائيات رقمية
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                summary_data.append({
                    'العمود': col,
                    'المتوسط': df[col].mean(),
                    'الوسيط': df[col].median(),
                    'الحد الأدنى': df[col].min(),
                    'الحد الأقصى': df[col].max()
                })
            
            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='الإحصائيات', index=False)
            
            # توزيع القيم النصية
            text_cols = df.select_dtypes(include=['object']).columns
            for i, col in enumerate(text_cols[:3]):  # أول 3 أعمدة نصية فقط
                value_counts = df[col].value_counts().head(10)
                value_counts.to_excel(writer, sheet_name=f'توزيع_{col}'[:31], index=True)
        
        return f"✅ تم التصدير بنجاح إلى: {filename}"
        
    except Exception as e:
        return f"❌ فشل في التصدير: {str(e)}"

def get_download_link(df, filename="البيانات.xlsx"):
    """إنشاء رابط تحميل للبيانات (لـ Streamlit)"""
    try:
        # تصدير إلى Excel في الذاكرة
        output = pd.ExcelWriter(filename, engine='openpyxl')
        df.to_excel(output, index=False)
        output.close()
        
        # قراءة الملف وتشفيره base64
        with open(filename, 'rb') as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">📥 تحميل البيانات</a>'
        return href
        
    except Exception as e:
        return f"❌ خطأ في إنشاء رابط التحميل: {str(e)}"

def format_percentage(value, decimals=1):
    """تنسيق النسب المئوية"""
    if pd.isna(value):
        return "غير متوفر"
    return f"{value:.{decimals}f}%"

def safe_divide(numerator, denominator, default=0):
    """قسمة آمة (تجنب القسمة على صفر)"""
    if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
        return default
    return numerator / denominator

def generate_summary_stats(df):
    """إنشاء إحصائيات ملخصة للبيانات"""
    if df is None or df.empty:
        return {"خطأ": "لا توجد بيانات"}
    
    stats = {
        'الأبعاد': {
            'عدد الصفوف': len(df),
            'عدد الأعمدة': len(df.columns)
        },
        'الأنواع': {},
        'القيم الفارغة': {},
        'القيم الفريدة': {}
    }
    
    for col in df.columns:
        stats['الأنواع'][col] = str(df[col].dtype)
        stats['القيم الفارغة'][col] = int(df[col].isna().sum())
        stats['القيم الفريدة'][col] = int(df[col].nunique())
    
    return stats

def clean_column_names(df):
    """تنظيف أسماء الأعمدة"""
    df_clean = df.copy()
    df_clean.columns = [
        str(col).strip().replace('\n', ' ').replace('\r', ' ')
        for col in df_clean.columns
    ]
    return df_clean

def detect_language(text):
    """كشف لغة النص (عربي/إنجليزي)"""
    if not isinstance(text, str):
        return "unknown"
    
    # كشف الحروف العربية
    arabic_chars = set('ابتثجحخدذرزسشصضطظعغفقكلمنهوي')
    text_chars = set(text)
    
    if any(char in arabic_chars for char in text_chars):
        return "arabic"
    else:
        return "english"

def log_message(message, level="INFO"):
    """تسجيل رسائل السجل"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {level}: {message}"
    print(formatted_message)
    return formatted_message

# دوال مساعدة للتواريخ
def convert_arabic_dates(date_str):
    """تحويل التواريخ العربية إلى الإنجليزية"""
    if not isinstance(date_str, str):
        return date_str
    
    arabic_to_english = {
        '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5',
        '٦': '6', '٧': '7', '٨': '8', '٩': '9', '٠': '0',
        '/': '/', '-': '-'
    }
    
    try:
        converted = ''.join(arabic_to_english.get(char, char) for char in date_str)
        return converted
    except:
        return date_str
