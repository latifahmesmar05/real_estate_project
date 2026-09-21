import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

#عنوان التطبيق
st.title("نموذج تقييم عقارات الرياض")
st.write("عرض استكشافي لبيانات العقارات و توقع الأسعار بإستخدام الذكاء الاصطناعي")

#قراءة البيانات
@st.cache_data
def load_data():
    df = pd.read_csv("riyadh_real_estate.csv.csv")
    
    df = df[['Neighborhood', 'Area (sqm)', 'Bedrooms', 'Selling Price (SAR)']]
    
    df = df.rename(columns={
        'Neighborhood': 'district',
        'Area (sqm)': 'size_sqm',
        'Bedrooms': 'bedrooms',
        'Selling Price (SAR)': 'price_sar'
    })
    
    df = df.dropna()
    
    return df

df = load_data()

#عرض جدول البيانات
st.subheader("نظرة عامة على البيانات")
st.dataframe(df)

# حسابات إحصائية بسيطة
st.subheader("إحصائيات سريعة")
col1, col2, col3 = st.columns(3)
col1.metric("عددالعقارات", len(df))
col2.metric("متوسط السعر", f"{df['price_sar'].mean():,.0f}ريال")
col3.metric("متوسط المساحة", f"{df['size_sqm'].mean():.0f}م²")

st.divider()

#رسوم بيانية تفاعلية
st.subheader("تحليلات و رسوم بيانية للبيانات")

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.write("**متوسط سعر العقار حسب الحي**")
    avg_price_district = df.groupby('district')['price_sar'].mean()
    st.bar_chart(avg_price_district)

with col_chart2:
    st.write("**العلاقة بين المساحة و السعر**")
    st.scatter_chart(data=df, x='size_sqm', y='price_sar', color='district')

#Machine Learning
st.subheader("توقع سعر العقار")

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#تجهيز المتغيرات المستقلة و التابعة
X = pd.get_dummies(df[['district', 'size_sqm', 'bedrooms']])
y = df['price_sar']

#تقسيم البيانات و تدريب النموذج
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#حساب التوقعات على البيانات الاختبار لقياس الاداء
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

#عرض مقاييس اداء النموذج
st.write("---")
st.subheader("اداء نموذج الذكاء الاصطناعي(Model Performance)")
col_metric1, col_metric2 = st.columns(2)
col_metric1.metric("دقة التفسير(R² Score)", f"{r2 * 100:.1f}%")
col_metric2.metric("متوسط هامش الخطأ(MAE)", f"{mae:,.0f}ريال")
st.write("---")

# مدخلات المستخدم للتقييم
st.write("أدخل مواصفات العقار لمعرفة السعر المتوقع:")

col_in1, col_in2, col_in3 = st.columns(3)
selected_district = col_in1.selectbox("اختر الحي", df['district'].unique())
input_size = col_in2.number_input("المساحة (م²)", min_value=50, max_value=2000, value=300)
input_bedrooms = col_in3.number_input("عدد الغرف", min_value=1, max_value=10, value=4)

# 7. زر التوقع والحساب
if st.button("💰 احسب السعر المتوقع"):

    # إنشاء صف جديد يحتوي على كافة الأعمدة بقيمة صفر
    input_df = pd.DataFrame(0, index=[0], columns=X.columns)
    
    # تعبئة القيم الأساسية
    input_df['size_sqm'] = input_size
    input_df['bedrooms'] = input_bedrooms
    
    # تحديد الحي المناسب
    district_column = f"district_{selected_district}"
    if district_column in input_df.columns:
        input_df[district_column] = 1

    # التوقع بواسطة النموذج
    predicted_price = model.predict(input_df)[0]
    
    # عرض النتيجة
    st.success(f"السعر التقديري المتوقع لهذا العقار هو: *{predicted_price:,.0f} ريال سعودي*")
    