import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# إضافة مسار src للمكتبات المخصصة
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_analyzer import AssetDataAnalyzer
from ai_predictor import AIPredictiveAssetAnalyzer

class AssetManagementApp:
    def __init__(self):
        self.set_page_config()
        
    def set_page_config(self):
        """إعدادات صفحة Streamlit"""
        st.set_page_config(
            page_title="نظام إدارة الأصول الذكي",
            page_icon="🏢",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # تخصيص التصميم
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5rem;
            color: #2e86ab;
            margin: 1rem 0;
            border-bottom: 2px solid #2e86ab;
            padding-bottom: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """تشغيل التطبيق الرئيسي"""
        st.markdown('<h1 class="main-header">🏢 نظام إدارة الأصول الذكي</h1>', 
                   unsafe_allow_html=True)
        
        # الشريط الجانبي
        self.sidebar()
        
        # المحتوى الرئيسي
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 تحميل البيانات", 
            "📈 التحليل المالي", 
            "🤖 الذكاء الاصطناعي", 
            "🗺️ الخرائط", 
            "⚙️ الإعدادات"
        ])
        
        with tab1:
            self.data_upload_tab()
        with tab2:
            self.financial_analysis_tab()
        with tab3:
            self.ai_analysis_tab()
        with tab4:
            self.maps_tab()
        with tab5:
            self.settings_tab()
    
    def sidebar(self):
        """الشريط الجانبي"""
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063512.png", 
                    width=100)
            
            st.markdown("### تحميل البيانات")
            uploaded_file = st.file_uploader(
                "اختر ملف Excel", 
                type=['xlsx', 'xls'],
                help="رفع ملف بيانات الأصول"
            )
            
            if uploaded_file is not None:
                if 'asset_data' not in st.session_state:
                    st.session_state.asset_data = uploaded_file
                st.success("✅ تم تحميل الملف بنجاح!")
            
            st.markdown("---")
            st.markdown("### خيارات سريعة")
            
            if st.button("🔄 تحديث البيانات"):
                st.rerun()
            
            if st.button("🧹 تنظيف الذاكرة"):
                self.clear_cache()
    
    def data_upload_tab(self):
        """تبويب تحميل البيانات"""
        st.markdown('<h2 class="section-header">📊 تحميل وتحليل البيانات</h2>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'asset_data' in st.session_state:
                try:
                    analyzer = AssetDataAnalyzer(st.session_state.asset_data)
                    if analyzer.load_data():
                        analyzer.clean_data()
                        st.subheader("معاينة البيانات")
                        st.dataframe(analyzer.df.head(100), use_container_width=True)
                        st.session_state.analyzer = analyzer
                        with col2:
                            self.show_basic_stats(analyzer)
                    else:
                        st.error("❌ فشل في تحميل البيانات")
                except Exception as e:
                    st.error(f"❌ خطأ في معالجة البيانات: {str(e)}")
            else:
                st.info("📁 يرجى تحميل ملف البيانات من الشريط الجانبي")
    
    def show_basic_stats(self, analyzer):
        st.subheader("📈 الإحصائيات الأساسية")
        basic_info = analyzer.get_basic_info()
        st.metric("إجمالي الأصول", f"{basic_info['إجمالي الأصول']:,}")
        st.metric("عدد الأعمدة", basic_info['عدد الأعمدة'])
    
    def financial_analysis_tab(self):
        st.markdown('<h2 class="section-header">📈 التحليل المالي والمحاسبي</h2>', 
                   unsafe_allow_html=True)
        st.info("🧾 سيتم تحميل التحليل المالي هنا لاحقًا...")
    
    def ai_analysis_tab(self):
        st.markdown('<h2 class="section-header">🤖 الذكاء الاصطناعي والتنبؤات</h2>', 
                   unsafe_allow_html=True)
        st.info("🤖 جاري تطوير وحدة الذكاء الاصطناعي...")
    
    def maps_tab(self):
        st.markdown('<h2 class="section-header">🗺️ التحليل الجغرافي</h2>', 
                   unsafe_allow_html=True)
        st.info("🗺️ سيتم عرض الخرائط التفاعلية هنا...")
    
    def settings_tab(self):
        st.markdown('<h2 class="section-header">⚙️ الإعدادات والتهيئة</h2>', 
                   unsafe_allow_html=True)
        st.info("⚙️ يمكنك تخصيص إعدادات النظام من هنا...")
    
    def clear_cache(self):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def main():
    app = AssetManagementApp()
    app.run()

if __name__ == "__main__":
    main()
