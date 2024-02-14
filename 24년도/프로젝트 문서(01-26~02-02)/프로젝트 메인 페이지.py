import streamlit as st

def page1():
    st.title('걸어서 세계속으로')
    st.write('설명과 웹캠')

def page2():
    st.title('주제2')
    st.write('설명')

page_select = st.radio('선택하세요', ['걸어서 세계속으로','주제2'])

if page_select == '걸어서 세계속으로 들어가기':
    page1()
elif page_select == '주제2로 들어가기':
    page2()