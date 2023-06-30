import streamlit as st
from streamlit_chat import message as st_message

from utils import get_completion_from_messages
from utils import convert_st_chat_to_openai_format


BOT_SETTINGS = """

Bạn là OrderBot, trợ lý ảo hỗ trợ nhận đơn hàng cho cửa hàng quần áo PandaML Shop. \

Trước tiên bạn cần chào khách hàng, sau đó hỏi khách hàng về mẫu quần áo mà họ lựa chọn và địa chỉ để giao hàng. \

Khi có đủ thông tin, bạn tóm tắt lại và hỏi khách hàng lần nữa xem họ muốn lấy thêm mẫu nào không trước khi \

tổng hợp lại đơn hàng cùng với giá tiền để khách hàng chốt đơn. \

Nhớ rằng bạn cần phải hỏi rõ thông tin để có thể xác định được đúng mẫu quần áo mà khách hàng lựa chọn. \

Shop chúng ta có những mẫu với giá cỡ S, M, L như sau (giá tiền theo trăm nghìn Việt Nam đồng) \
A1 100, 150, 200 \
A2 100, 160, 300 \
A3 80, 100, 120 \

"""


def generate_answer():
    # get user message and add to chat history 
    user_message = st.session_state.input_text
    st.session_state.history.append({'message': user_message, 'is_user': True})
    # convert to OpenAI format
    messages = convert_st_chat_to_openai_format(st.session_state.history)
    messages = [{'role': 'system', 'content': BOT_SETTINGS}] + messages
    # get bot message and add to chat history
    bot_message = get_completion_from_messages(messages)
    st.session_state.history.append({'message': bot_message, 'is_user': False})
    # clear input text
    st.session_state['input_text'] = ''


# initialize 
if 'history' not in st.session_state:
    st.session_state.history = []

# define chatbot interface
st.title('OrderBot - Trợ lý ảo nhận đơn hàng')
st.text_input('Chat với tôi', key='input_text', on_change=generate_answer)

# display chat message
for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i))
