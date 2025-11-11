import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.ensemble import IsolationForest
import joblib
import warnings
warnings.filterwarnings('ignore')

class AIPredictiveAssetAnalyzer:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        
    def prepare_ai_features(self):
        """تحضير البيانات للذكاء الاصطناعي"""
        if self.analyzer.df is None:
            raise ValueError("لم يتم تحميل البيانات بعد")
        
        df = self.analyzer.df.copy()
        
        # تحويل التواريخ
        date_col = 'Date Placed in Service' if 'Date Placed in Service' in df.columns else 'تاريخ الدخول في الخدمة'
        if date_col in df.columns:
            df['Date_Placed_Service'] = pd.to_datetime(df[date_col], errors='coerce')
            df['Asset_Age_Days'] = (pd.Timestamp('2023-12-30') - df['Date_Placed_Service']).dt.days
            df['Asset_Age_Years'] = df['Asset_Age_Days'] / 365.25
        else:
            df['Asset_Age_Years'] = 3  # قيمة افتراضية
        
        # معالجة الأعمدة الرقمية
        cost_col = 'Cost' if 'Cost' in df.columns else 'التكلفة'
        depreciation_col = 'Depreciation amount' if 'Depreciation amount' in df.columns else 'قسط الاهلاك'
        nbv_col = 'Net Book Value' if 'Net Book Value' in df.columns else 'القيمة الدفترية'
        life_col = 'Useful Life' if 'Useful Life' in df.columns else 'العمر الإنتاجي'
        
        numeric_columns = [cost_col, depreciation_col, nbv_col, life_col]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # معالجة النصوص باستخدام Label Encoding
        text_columns = ['Level 1 FA Module - English Description', 
                       'Level 2 FA Module - English Description',
                       'Level 3 FA Module - English Description',
                       'Manufacturer', 'Region', 'City']
        
        for col in text_columns:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(
                    df[col].fillna('Unknown')
                )
        
        # ميزات إضافية
        df['Cost_per_Year'] = df[cost_col] / df[life_col].replace(0, 1)
        df['Depreciation_Rate'] = df[depreciation_col] / df[cost_col].replace(0, 1)
        
        # تحديد الميزات النهائية
        self.feature_columns = [
            cost_col, 'Asset_Age_Years', life_col, 'Cost_per_Year', 'Depreciation_Rate'
        ]
        
        # إضافة الأعمدة المشفرة إذا كانت موجودة
        for col in text_columns:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                self.feature_columns.append(encoded_col)
        
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        self.ai_ready_data = df[self.feature_columns + [nbv_col, depreciation_col]].dropna()
        
        print(f"✅ تم تحضير {len(self.ai_ready_data)} عينة للذكاء الاصطناعي")
        return self.ai_ready_data

    def predict_asset_depreciation(self):
        """تنبؤ بقيمة الإهلاك المستقبلية باستخدام Random Forest"""
        try:
            data = self.prepare_ai_features()
            
            if len(data) < 10:
                return {"خطأ": "لا توجد بيانات كافية للتدريب (يحتاج 10 عينات على الأقل)"}
            
            X = data[self.feature_columns]
            y = data['Depreciation amount' if 'Depreciation amount' in data.columns else 'قسط الاهلاك']
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # تدريب النموذج
            self.models['depreciation_rf'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['depreciation_rf'].fit(X_train, y_train)
            
            # التنبؤ والتقييم
            y_pred = self.models['depreciation_rf'].predict(X_test)
            
            results = {
                'model': 'Random Forest',
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'samples_used': len(data),
                'feature_importance': dict(zip(self.feature_columns, 
                                             self.models['depreciation_rf'].feature_importances_))
            }
            
            return results
            
        except Exception as e:
            return {"خطأ": f"فشل في تنبؤ الإهلاك: {str(e)}"}

    def predict_asset_failure_risk(self):
        """تنبؤ بمخاطر تعطل الأصول"""
        try:
            data = self.prepare_ai_features()
            
            if len(data) < 10:
                return {"خطأ": "لا توجد بيانات كافية لتحليل المخاطر"}
            
            # تعريف مخاطر التعطل بناءً على العمر ومعدل الإهلاك
            data['Failure_Risk'] = np.where(
                (data['Asset_Age_Years'] > data['Useful Life'] * 0.8) | 
                (data['Depreciation_Rate'] > 0.4), 'High',
                np.where(
                    (data['Asset_Age_Years'] > data['Useful Life'] * 0.6) |
                    (data['Depreciation_Rate'] > 0.25), 'Medium', 'Low'
                )
            )
            
            risk_counts = data['Failure_Risk'].value_counts().to_dict()
            
            return {
                'risk_distribution': risk_counts,
                'total_assets_analyzed': len(data),
                'high_risk_percentage': (risk_counts.get('High', 0) / len(data)) * 100
            }
            
        except Exception as e:
            return {"خطأ": f"فشل في تحليل المخاطر: {str(e)}"}

    def asset_clustering_analysis(self, n_clusters=4):
        """تجميع الأصول باستخدام خوارزميات Clustering"""
        try:
            data = self.prepare_ai_features()
            
            if len(data) < n_clusters:
                return {"خطأ": f"لا توجد بيانات كافية للتجميع (يحتاج {n_clusters} عينات على الأقل)"}
            
            # تحديد الميزات للتجميع
            cluster_features = ['Cost', 'Asset_Age_Years', 'Depreciation_Rate', 'Cost_per_Year']
            cluster_features = [f for f in cluster_features if f in data.columns]
            
            if len(cluster_features) < 2:
                return {"خطأ": "لا توجد ميزات كافية للتجميع"}
            
            cluster_data = data[cluster_features].dropna()
            
            # تطبيع البيانات
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # تطبيق K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # تحليل العناقيد
            cluster_data['Cluster'] = clusters
            cluster_analysis = cluster_data.groupby('Cluster').agg({
                'Cost': ['mean', 'count'],
                'Asset_Age_Years': 'mean',
                'Depreciation_Rate': 'mean'
            }).round(2)
            
            self.models['kmeans'] = kmeans
            self.scalers['clustering'] = scaler
            
            return {
                'clusters': cluster_analysis.to_dict(),
                'cluster_labels': clusters.tolist(),
                'inertia': kmeans.inertia_,
                'n_clusters': n_clusters
            }
            
        except Exception as e:
            return {"خطأ": f"فشل في تجميع الأصول: {str(e)}"}

    def anomaly_detection(self):
        """كشف الشذوذ في بيانات الأصول"""
        try:
            data = self.prepare_ai_features()
            
            if len(data) < 10:
                return {"خطأ": "لا توجد بيانات كافية لكشف الشذوذ"}
            
            # استخدام Isolation Forest لكشف الشذوذ
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(data[self.feature_columns])
            
            data['Is_Anomaly'] = anomalies
            anomalies_count = (anomalies == -1).sum()
            
            self.models['anomaly_detector'] = iso_forest
            
            if anomalies_count > 0:
                anomaly_data = data[data['Is_Anomaly'] == -1]
                return {
                    'anomalies_count': int(anomalies_count),
                    'anomaly_percentage': (anomalies_count / len(data)) * 100,
                    'total_samples': len(data),
                    'message': f"تم اكتشاف {anomalies_count} حالة شذوذ"
                }
            else:
                return {
                    'anomalies_count': 0,
                    'anomaly_percentage': 0.0,
                    'total_samples': len(data),
                    'message': "لم يتم اكتشاف شذوذ في البيانات"
                }
                
        except Exception as e:
            return {"خطأ": f"فشل في كشف الشذوذ: {str(e)}"}

    def optimal_replacement_timing(self):
        """تحديد التوقيت الأمثل لاستبدال الأصول"""
        try:
            data = self.prepare_ai_features()
            
            recommendations = []
            for idx, row in data.iterrows():
                age = row['Asset_Age_Years']
                useful_life = row['Useful Life']
                depreciation_rate = row['Depreciation_Rate']
                
                # تحليل التكلفة والعائد
                if age > useful_life * 0.8:
                    recommendation = 'استبدال عاجل'
                elif age > useful_life * 0.6 and depreciation_rate > 0.3:
                    recommendation = 'استبدال قريب'
                elif age > useful_life * 0.5:
                    recommendation = 'صيانة استباقية'
                else:
                    recommendation = 'صيانة دورية'
                
                recommendations.append(recommendation)
            
            rec_counts = pd.Series(recommendations).value_counts().to_dict()
            
            return {
                'recommendation_distribution': rec_counts,
                'total_analyzed': len(recommendations),
                'urgent_replacement_percentage': (rec_counts.get('استبدال عاجل', 0) / len(recommendations)) * 100
            }
            
        except Exception as e:
            return {"خطأ": f"فشل في تحليل توقيت الاستبدال: {str(e)}"}

    def save_ai_models(self, filepath='asset_ai_models.pkl'):
        """حفظ النماذج المدربة"""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns
            }
            
            joblib.dump(model_data, filepath)
            return f"✅ تم حفظ النماذج في {filepath}"
            
        except Exception as e:
            return f"❌ فشل في حفظ النماذج: {str(e)}"

    def load_ai_models(self, filepath='asset_ai_models.pkl'):
        """تحميل النماذج المدربة"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers'] 
            self.encoders = model_data['encoders']
            self.feature_columns = model_data['feature_columns']
            
            return "✅ تم تحميل النماذج بنجاح"
            
        except Exception as e:
            return f"❌ فشل في تحميل النماذج: {str(e)}"

    def generate_ai_report(self):
        """تقرير شامل للذكاء الاصطناعي"""
        report = {
            'depreciation_prediction': self.predict_asset_depreciation(),
            'failure_risk': self.predict_asset_failure_risk(),
            'clustering': self.asset_clustering_analysis(),
            'anomaly_detection': self.anomaly_detection(),
            'replacement_timing': self.optimal_replacement_timing()
        }
        
        return report
