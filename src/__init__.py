"""
نظام إدارة الأصول الذكي
وحدة تحليل البيانات والذكاء الاصطناعي
"""

__version__ = "1.0.0"
__author__ = "نظام إدارة الأصول"
__description__ = "نظام متكامل لإدارة الأصول الثابتة باستخدام الذكاء الاصطناعي"

# استيراد الفئات الرئيسية لتكون متاحة مباشرة
from .data_analyzer import AssetDataAnalyzer
from .ai_predictor import AIPredictiveAssetAnalyzer
from .utils import format_currency, validate_data, export_to_excel

# قائمة بما يمكن استيراده
__all__ = [
    'AssetDataAnalyzer',
    'AIPredictiveAssetAnalyzer', 
    'format_currency',
    'validate_data',
    'export_to_excel'
]

print(f"✅ تم تحميل {__name__} بنسخة {__version__}")
