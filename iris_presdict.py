# import numpy as np
# import pandas as pd
import streamlit as st

ps = st.navigation(
    [
        st.Page("pages/page1.py", title="IRIS品種預測"), # path, title, 圖示
        st.Page("pages/page2.py", title="IRIS樣本分布圖")
    ]
)

ps.run()