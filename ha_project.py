# -*- coding: utf-8 -*-
"""HA project.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g4lzNyz0dVl39HsAE0iT675PbE4wnLWm
"""

! pip install streamlit -q

!wget -q -O - ipv4.icanhazip.com

! streamlit run app.py & npx localtunnel --port 8501