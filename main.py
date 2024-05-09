import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
import time

from KNN import knn_predict, generate_data, add_noise, knn_predict_manhattan, knn_predict_chebyshev

st.title("KNN分类可视化")

n_samples = st.sidebar.number_input("样本数量", 5, 1000000, 1000)
test_samples = st.sidebar.number_input("测试样本数量", 5, 1000000, 1000)
n_noise = st.sidebar.number_input("噪声数量", 0, 1000000, 0)
k = st.sidebar.number_input("K值", 1, 100, 5)

generate_button = st.sidebar.button("生成数据")

if generate_button:
    X, Y = generate_data(n_samples)
    if n_noise > 0:
        X, Y = add_noise(X, Y, n_noise)
    st.session_state["X"] = X
    st.session_state["Y"] = Y

if "X" in st.session_state and "Y" in st.session_state:
    fig, ax = plt.subplots()
    for i in range(1, 10):
        ax.scatter(st.session_state["X"][st.session_state["Y"] == i][:, 0], st.session_state["X"][st.session_state["Y"] == i][:, 1], label=f"Class {i}")
    st.pyplot(fig)

    if st.button("分类"):
        X_test, Y_test = generate_data(test_samples)
        start_time = time.time()
        Y_pred = knn_predict(st.session_state["X"], st.session_state["Y"], X_test, k)
        print(time.time()-start_time)

        # 计算错误率
        error_rate = np.mean(Y_pred != Y_test)
        st.write(f"错误率: {error_rate:.2%}")

        fig, ax = plt.subplots()
        for i in range(1, 10):
            ax.scatter(X_test[Y_pred == i][:, 0], X_test[Y_pred == i][:, 1], label=f"Class {i}")
        st.pyplot(fig)
