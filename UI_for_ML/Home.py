import streamlit as st

def main():
    st.markdown("""
        <div style="text-align: center; margin-top: 20%;">
            <h1 style="font-family: 'Arial Black', sans-serif; color: #6C3BAA; font-size: 3em; text-shadow: 2px 2px 4px #000000;">
            Diabetes Predictor
            </h1>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Enter to The lab", key="enter_lab"):
            st.switch_page("pages/DB.py")

if __name__ == "__main__":
    main()
