import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page, chat_box
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
import os
import sys
from configs import VERSION
from server.utils import api_address


api = ApiRequest(base_url=api_address())

if 'language' not in st.session_state:
    st.session_state['language'] = 'English' # default language setting

def update_language_choice():
    st.session_state['language'] = st.session_state['selected_language']

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "Langchain-Chatchat WebUI",
        os.path.join("img", "chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/chatchat-space/Langchain-Chatchat',
            'Report a bug': "https://github.com/chatchat-space/Langchain-Chatchat/issues",
            'About': f"""欢迎使用 Langchain-Chatchat WebUI {VERSION}！"""
        }
    )

    pages_text = {
        "dialogue":{
            "简体中文": "对话",
            "English": "Dialogue"
        },
        "kb":{
            "简体中文": "知识库管理",
            "English": "Manage Knowledge Base"
        }
    }

    pages = {
        pages_text["dialogue"][st.session_state['language']]: {
            "icon": "chat",
            "func": dialogue_page,
        },
        pages_text["kb"][st.session_state['language']]: {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "logo-long-chatchat-trans-v2.png"
            ),
            use_column_width=True
        )
        
        version_text = {
            "简体中文": "当前版本：",
            "English": "Current Version: "
        }
        st.caption(
            f"""<p align="right">{version_text[st.session_state['language']]}{VERSION}</p>""",
            unsafe_allow_html=True,
        )

        select_language_text = {
            'English': '🌐 Interface Display Language',
            '简体中文': '🌐 界面显示语言',
        }

        language_options = ["简体中文", "English"]
        
        selected_language = st.selectbox(
            select_language_text[st.session_state['language']],
            options=language_options,
            index=language_options.index(st.session_state.get('language', '简体中文')),
            on_change=update_language_choice,
            key='selected_language'  
        )

        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
