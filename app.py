import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 1. CẤU HÌNH TRANG (LỆNH ĐẦU TIÊN)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Giao dịch Gian lận",
    page_icon="🛡️"
)

# ==========================================
# 2. HÀM CACHE NẠP DỮ LIỆU DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_bytes)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# Khởi tạo danh sách các biến đặc trưng dựa theo kiến trúc của notebook
FEATURES = [f"X_{i}" for i in range(1, 15)]
TARGET = "default"

# ==========================================
# 3. THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu huấn luyện mẫu
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện mẫu (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn file dataset1.csv hoặc cấu trúc tương đương chứa các cột X_1 đến X_14 và cột mục tiêu 'default'."
    )
    
    st.divider()
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: Random Forest Classifier")
    
    # Các siêu tham số trích xuất và tối ưu từ notebook
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, max_value=300, value=100, step=10,
        help="Số lượng cây quyết định trong rừng."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=1, max_value=30, value=15, step=1,
        help="Độ sâu tối đa của mỗi cây quyết định (None nếu không giới hạn)."
    )
    
    min_samples_split = st.slider(
        "Mẫu tối thiểu để tách nút (min_samples_split)", 
        min_value=2, max_value=20, value=2, step=1,
        help="Số lượng mẫu tối thiểu cần thiết để phân tách một nút nội bộ."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        value=42, step=1,
        help="Giá trị seed để đảm bảo kết quả huấn luyện có thể tái lặp."
    )
    
    st.divider()
    # NÚT HÀNH ĐỘNG DUY NHẤT KÍCH HOẠT HUẤN LUYỆN
    trigger_train = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu quá trình phân tách dữ liệu và huấn luyện Random Forest."
    )

# ==========================================
# 4. THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Ứng dụng Phát hiện Giao dịch Gian lận")
st.caption("Giải pháp Học máy hỗ trợ nhận diện và phân loại sớm các giao dịch rủi ro/gian lận tài chính dựa trên mô hình học máy Random Forest.")

if uploaded_file is None:
    st.info("👋 Chào mừng bạn! Vui lòng tải tệp dữ liệu ở thanh Sidebar bên trái để bắt đầu khám phá và huấn luyện mô hình.")
    st.stop()
else:
    # Đọc dữ liệu thô qua hàm cache đã khai báo
    df_raw = load_data(uploaded_file, uploaded_file.name)
    if df_raw is None:
        st.error("Không thể đọc được cấu trúc dữ liệu. Vui lòng kiểm tra lại định dạng file.")
        st.stop()
        
    st.caption(f"📁 **Đang dùng tệp:** {uploaded_file.name} | Chiều dữ liệu: {df_raw.shape[0]} dòng, {df_raw.shape[1]} cột.")
st.divider()

# ==========================================
# 5. KHỐI XỬ LÝ HUẤN LUYỆN (LƯU SESSION STATE)
# ==========================================
if trigger_train:
    with st.spinner("🔄 Hệ thống đang xử lý dữ liệu và huấn luyện mô hình..."):
        # Kiểm tra xem có đủ các cột đặc trưng và cột mục tiêu không
        missing_cols = [col for col in FEATURES + [TARGET] if col not in df_raw.columns]
        if missing_cols:
            st.error(f"Dữ liệu tải lên thiếu các cột bắt buộc sau: {missing_cols}")
        else:
            X = df_raw[FEATURES]
            y = df_raw[TARGET]
            
            # Chia tập Train/Test theo tỷ lệ của notebook (80/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
            
            # Khởi tạo mô hình theo tham số cấu hình trên UI
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            
            # Dự báo trên tập kiểm định để lấy chỉ số đánh giá
            y_pred = model.predict(X_test)
            
            # Tính toán các chỉ tiêu đo lường hiệu suất
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist()
            }
            
            # Lưu trữ vào session_state để tái sử dụng ở các tab mà không cần train lại
            st.session_state["trained_model"] = model
            st.session_state["metrics"] = metrics
            st.session_state["features_columns"] = FEATURES
            st.success("🎉 Huấn luyện thành công! Chuyển sang tab 'Kết quả huấn luyện' để xem chi tiết.")

# ==========================================
# 6. KHỞI TẠO CÁC TABS GIAO DIỆN CHÍNH
# ==========================================
tab_summary, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa", 
    "🎯 Kết quả huấn luyện & Kiểm định", 
    "🔮 Sử dụng mô hình"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab_summary:
    st.subheader("Phân tích cấu trúc dữ liệu thô")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng bản ghi (Dòng)", f"{df_raw.shape[0]:,}")
    with col_m2:
        st.metric("Số lượng thuộc tính (Cột)", f"{df_raw.shape[1]:,}")
    with col_m3:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
        
    st.write("### 🔎 Hiển thị 5 dòng dữ liệu đầu tiên")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.write("### 📐 Bảng thống kê mô tả các biến đặc trưng đưa vào mô hình")
    # Chỉ thống kê mô tả các biến được đưa vào bài toán ML
    available_model_cols = [c for c in FEATURES + [TARGET] if c in df_raw.columns]
    if available_model_cols:
        st.dataframe(df_raw[available_model_cols].describe(), use_container_width=True)
    else:
        st.warning("Không tìm thấy các cột đặc trưng tiêu chuẩn của hệ thống.")

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab_viz:
    st.subheader("Trực quan xu hướng và phân phối dữ liệu")
    
    # Ưu tiên hiển thị biến mục tiêu trước (bài toán phân loại giám sát)
    if TARGET in df_raw.columns:
        st.write("#### Phân phối nhãn mục tiêu (0: Bình thường, 1: Gian lận)")
        target_counts = df_raw[TARGET].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: 'Bình thường (0)', 1: 'Gian lận (1)'})
        
        fig_target = px.bar(
            target_counts, x='Trạng thái', y='Số lượng', 
            color='Trạng thái', text_auto=True,
            color_discrete_map={'Bình thường (0)': '#2ecc71', 'Gian lận (1)': '#e74c3c'}
        )
        fig_target.update_layout(height=350)
        st.plotly_chart(fig_target, use_container_width=True)
    
    st.write("#### Trực quan hóa phân phối phân vị các biến đầu vào")
    # Cho phép người dùng tùy chọn chọn nhóm biến để vẽ dạng lưới 2x2 để tránh quá tải
    selected_features = st.multiselect(
        "Chọn tối đa 4 thuộc tính để hiển thị biểu đồ phân phối:",
        options=FEATURES,
        default=FEATURES[:4],
        max_selections=4
    )
    
    if selected_features:
        # Bố trí dạng lưới 2x2 bằng 2 hàng cột chéo nhau
        rows_cols = st.columns(2)
        for idx, col_name in enumerate(selected_features):
            current_col = rows_cols[idx % 2]
            with current_col:
                st.write(f"**Phân phối của thuộc tính {col_name}**")
                fig_hist = px.histogram(
                    df_raw, x=col_name, color=TARGET if TARGET in df_raw.columns else None,
                    marginal="box", barmode="overlay",
                    color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                )
                fig_hist.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Vui lòng lựa chọn ít nhất một thuộc tính để xem biểu đồ.")

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------
with tab_metrics:
    st.subheader("Đánh giá độ chính xác của Mô hình toán học")
    
    # Kiểm tra xem mô hình đã được huấn luyện thành công ở Sidebar chưa
    if "metrics" not in st.session_state:
        st.info("💡 Chưa tìm thấy dữ liệu huấn luyện. Vui lòng thiết lập cấu hình tham số ở Sidebar bên trái và ấn nút **'Huấn luyện mô hình'**.")
    else:
        res = st.session_state["metrics"]
        
        # Hiển thị các chỉ số đo lường vô hướng hàng đầu qua st.metric
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Độ chính xác chung (Accuracy)", f"{res['accuracy']:.4f}")
        with m_col2:
            st.metric("Độ chính xác xác định (Precision)", f"{res['precision']:.4f}")
        with m_col3:
            st.metric("Tỷ lệ bỏ sót (Recall)", f"{res['recall']:.4f}")
        with m_col4:
            st.metric("Chỉ số F1-Score cân bằng", f"{res['f1_score']:.4f}")
            
        st.divider()
        
        # Biểu diễn trực quan Ma trận nhầm lẫn (Confusion Matrix)
        st.write("#### 🧱 Ma trận nhầm lẫn (Confusion Matrix)")
        cm_data = np.array(res["confusion_matrix"])
        x_axis = ['Dự báo Bình thường (0)', 'Dự báo Gian lận (1)']
        y_axis = ['Thực tế Bình thường (0)', 'Thực tế Gian lận (1)']
        
        fig_cm = ff.create_annotated_heatmap(
            cm_data, x=x_axis, y=y_axis, 
            colorscale='YlOrRd', showscale=True
        )
        fig_cm.update_layout(margin=dict(t=40, b=40))
        st.plotly_chart(fig_cm, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH"
# ------------------------------------------
with tab_inference:
    st.subheader("Vận hành dự báo giao dịch rủi ro trực tuyến")
    
    if "trained_model" not in st.session_state:
        st.info("💡 Chức năng dự báo yêu cầu mô hình phải được huấn luyện trước. Hãy hoàn tất bước ấn nút tại Sidebar.")
    else:
        current_model = st.session_state["trained_model"]
        
        # Cho phép người dùng chuyển đổi linh hoạt 2 chế độ dự đoán dữ liệu mới
        mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào phục vụ dự đoán:",
            options=["Nhập chỉ số trực tiếp bằng Form", "Tải tệp danh sách hàng loạt (X_test)"],
            horizontal=True
        )
        
        # CHẾ ĐỘ 1 — NHẬP TRỰC TIẾP
        if mode == "Nhập chỉ số trực tiếp bằng Form":
            st.write("#### 📝 Điền các thông số kỹ thuật của giao dịch hiện tại")
            
            with st.form("single_prediction_form"):
                form_cols = st.columns(3)
                input_data = {}
                
                # Tạo động các ô nhập liệu số dựa trên dải giá trị thực tế của dataset1
                for idx, col_name in enumerate(FEATURES):
                    # Tính toán giá trị trung vị mặc định dựa trên dữ liệu lịch sử thô đã nạp
                    default_val = float(df_raw[col_name].median()) if col_name in df_raw.columns else 0.0
                    min_val = float(df_raw[col_name].min()) if col_name in df_raw.columns else -100.0
                    max_val = float(df_raw[col_name].max()) if col_name in df_raw.columns else 100.0
                    
                    target_form_col = form_cols[idx % 3]
                    with target_form_col:
                        input_data[col_name] = st.number_input(
                            f"Nhập thuộc tính {col_name}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            format="%.6f",
                            help=f"Giá trị thực tế trong tập mẫu dao động từ {min_val:.2f} tới {max_val:.2f}."
                        )
                
                submit_predict = st.form_submit_button("🔍 Phân tích rủi ro giao dịch", type="primary")
                
                if submit_predict:
                    # Chuyển đổi dữ liệu form sang DataFrame có cấu trúc cột chuẩn xác
                    single_df = pd.DataFrame([input_data])[FEATURES]
                    prediction = current_model.predict(single_df)[0]
                    
                    try:
                        probabilities = current_model.predict_proba(single_df)[0]
                        fraud_prob = probabilities[1]
                    except:
                        fraud_prob = None
                    
                    st.divider()
                    if prediction == 1:
                        st.error(f"🚨 **CẢNH BÁO:** Hệ thống nhận diện đây là một giao dịch **GIAN LẬN** / **RỦI RO CAO**.")
                        if fraud_prob is not None:
                            st.metric("Xác suất phân loại rủi ro", f"{fraud_prob * 100:.2f} %")
                    else:
                        st.success(f"✅ **AN TOÀN:** Giao dịch được thẩm định ở trạng thái **BÌNH THƯỜNG**.")
                        if fraud_prob is not None:
                            st.metric("Xác suất an toàn", f"{(1 - fraud_prob) * 100:.2f} %")
                            
        # CHẾ ĐỘ 2 — TẢI FILE THEO CẤU TRÚC X_TEST
        elif mode == "Tải tệp danh sách hàng loạt (X_test)":
            st.write("#### 📂 Tải lên danh sách dữ liệu mới cần quét rủi ro số lượng lớn")
            new_file_uploader = st.file_uploader(
                "Tải lên tệp danh sách cần dự báo (.csv, .xlsx)", 
                type=["csv", "xlsx"],
                key="inference_file_uploader"
            )
            
            if new_file_uploader is not None:
                new_df = load_data(new_file_uploader, new_file_uploader.name)
                
                if new_df is not None:
                    # Kiểm tra cấu trúc phân bố cột (Schema validation)
                    missing_features = [col for col in FEATURES if col not in new_df.columns]
                    
                    if missing_features:
                        st.error(f"Tệp tin tải lên không hợp lệ. Bản ghi bị thiếu các cột chỉ số bắt buộc: {missing_features}")
                    else:
                        # Thực hiện quét dự báo đồng loạt mà không thay đổi thứ tự hàng
                        input_x = new_df[FEATURES]
                        batch_preds = current_model.predict(input_x)
                        
                        # Đính kèm thêm cột kết quả vào dataframe hiển thị
                        output_df = new_df.copy()
                        output_df["Kết_Luận_Dự_Báo"] = batch_preds
                        output_df["Trạng_Thái_Ý_Nghĩa"] = output_df["Kết_Luận_Dự_Báo"].map({0: "Bình thường", 1: "Nguy cơ Gian lận"})
                        
                        st.success(f"Quét thành công! Đã xử lý {output_df.shape[0]} dòng dữ liệu giao dịch mới.")
                        
                        # Đếm thống kê nhanh số lượng phân loại rủi ro mới phát hiện
                        fraud_count = int((batch_preds == 1).sum())
                        st.warning(f"⚠️ Phát hiện khẩn cấp **{fraud_count}** giao dịch có dấu hiệu bất thường/gian lận trong file dữ liệu.")
                        
                        # Hiển thị bảng dữ liệu kết quả tích hợp
                        st.write("##### 📊 Bảng tổng hợp kết quả phân tích chi tiết:")
                        st.dataframe(output_df, use_container_width=True)
                        
                        # Cho phép người dùng tải tệp kết quả đầu ra định dạng CSV chuẩn
                        csv_data = output_df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name="Ket_qua_du_bao_giao_dich.csv",
                            mime="text/csv"
                        )
