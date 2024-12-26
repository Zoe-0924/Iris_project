import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("IRIS樣本分布圖")

df = pd.read_csv("iris.csv")
fig,ax = plt.subplots()
mapping = {"Setosa":0, "Versicolor":1, "Virginica":2}
color = ['red', 'green', 'blue']

tab1, tab2 = st.tabs(["依花萼長寬繪圖", "依花瓣長寬繪圖"])

with tab1:
    # st.write("222")
    fig, ax = plt.subplots()
    for i,j in mapping.items():
        subset = df[df["variety"]==i]
        ax.scatter(subset["sepal.length"], subset["sepal.width"], label=i)
    ax.set_xlabel("sepal.length")
    ax.set_ylabel("sepal.width")
    ax.legend()
    st.pyplot(fig)

with tab2:
    # st.write("222")
    fig2, ax2 = plt.subplots()
    for i,j in mapping.items():
        subset = df[df["variety"]==i]
        ax2.scatter(subset["petal.length"], subset["petal.width"], label=i)
    ax2.set_xlabel("petal.length")
    ax2.set_ylabel("petal.width")
    ax2.legend()
    st.pyplot(fig2)