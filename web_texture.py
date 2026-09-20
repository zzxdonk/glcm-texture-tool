import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import streamlit as st
import pandas as pd

# 兼容中文路径读图函数
def cv2_imread_chinese(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def process_image(img_bytes):
    img = cv2_imread_chinese(img_bytes)
    if img is None:
        raise Exception("图片读取失败，文件损坏或格式不支持")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 基础统计
    mean_gray = np.mean(gray)
    var_gray = np.var(gray)
    std_gray = np.std(gray)
    min_gray = np.min(gray)
    max_gray = np.max(gray)
    entropy_img = shannon_entropy(gray)

    # GLCM纹理
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')
    homogeneity = graycoprops(glcm, 'homogeneity')

    angle_names = ["0°", "45°", "90°", "135°"]
    stat_data = {
        "灰度均值": mean_gray,
        "灰度方差": var_gray,
        "灰度标准差": std_gray,
        "最小灰度": min_gray,
        "最大灰度": max_gray,
        "香农熵": entropy_img
    }
    tex_df = pd.DataFrame({
        "角度": angle_names,
        "对比度": contrast[0],
        "相关性": correlation[0],
        "能量": energy[0],
        "同质性": homogeneity[0]
    })
    return stat_data, tex_df, gray

# ----------------网页界面----------------
st.title("图像GLCM纹理特征计算工具")
upload_file = st.file_uploader("上传图片（jpg/png/tif）", type=["jpg","png","jpeg","tif","tiff"])

if upload_file is not None:
    img_bytes = upload_file.read()
    img = cv2_imread_chinese(img_bytes)
    st.image(img, channels="BGR", caption="上传图像")
    try:
        stat_result, texture_df, gray_img = process_image(img_bytes)
        st.subheader("基础灰度统计指标")
        st.write(stat_result)
        st.subheader("GLCM纹理特征结果")
        st.dataframe(texture_df)
        # 导出csv
        csv = texture_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("下载纹理结果CSV", csv, "texture_result.csv", mime="text/csv")
    except Exception as e:
        st.error(f"处理失败：{str(e)}")
