import google.generativeai as genai
import streamlit as st

GOOGLE_API_KEY=""
genai.configure(api_key=GOOGLE_API_KEY)

geminiModel=genai.GenerativeModel("gemini-pro")


#simple code
# response=geminiModel.generate_content("Tell me a joke")
# print(response.text)


#below code is just for question and answer no history or any relevant info
# def get_gemini_response(question):
#     response=geminiModel.generate_content(question)
#     return response.text

# st.set_page_config(page_title="Q&a Demo")
# st.header("Gemini LLM")


# input=st.text_input("Input: ",key="input")
# submit=st.button("Ask the question")

# if submit:
#     response=get_gemini_response(input)
#     st.subheader("The response is")
#     st.write(response)


#below code is for chat history and relevant info

chat=geminiModel.start_chat(history=[]) #only when you want to have the history 

def get_gemini_response(question):
    response=chat.send_message(question,stream=True)
    return response

st.set_page_config(page_title="Q&a Demo")
st.header("Gemini LLM Application")


if 'chat_history' not in st.session_state:
    st.session_state['chat_history']=[]

input=st.text_input("Input: ",key="input")
submit=st.button("Ask the question")

if submit and input:
    response=get_gemini_response(input)
    st.session_state['chat_history'].append({"You",input})
    st.subheader("The Response is")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append({"Bot",chunk.text})

st.subheader("The Chat history is:")

for role,text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")