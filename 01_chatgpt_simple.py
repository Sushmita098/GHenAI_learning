import openai
import gradio
import streamlit as st

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = ""  # Replace with your Azure OpenAI endpoint
AZURE_OPENAI_API_KEY = ""  # Replace with your API key
AZURE_DEPLOYMENT_NAME = "gpt-4o"  # Replace with your deployed model name

# Set up the OpenAI API client with Azure-specific configurations
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2024-02-01"  # Use the latest API version
openai.api_key = AZURE_OPENAI_API_KEY


#step 1
# Call GPT-4o for a completion
# response = openai.ChatCompletion.create(
#     engine=AZURE_DEPLOYMENT_NAME,  # Use "engine" instead of "model" for Azure OpenAI
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Tell me things I can build using airflow and genai."}
#     ],
#     max_tokens=1000
# )

# # Print the response
# print(response["choices"][0]["message"]["content"])



#step 2
# messages = []
# system_msg = input("What type of chatbot would you like to create?\n")
# messages.append({"role": "system", "content": system_msg})

# print("Your new assistant is ready!")
# while input != "quit()":
#     message = input()
#     messages.append({"role": "user", "content": message})
#     response = openai.ChatCompletion.create(
#         engine=AZURE_DEPLOYMENT_NAME,
#         messages=messages)
#     reply = response["choices"][0]["message"]["content"]
#     messages.append({"role": "assistant", "content": reply})
#     print("\n" + reply + "\n")


#step 3 web app using gradio
# messages = [{"role": "system", "content": "You are a financial experts that specializes in real estate investment and negotiation"}]

# def CustomChatGPT(user_input):
#     messages.append({"role": "user", "content": user_input})
#     response = openai.ChatCompletion.create(
#         engine=AZURE_DEPLOYMENT_NAME,
#         messages = messages
#     )
#     ChatGPT_reply = response["choices"][0]["message"]["content"]
#     messages.append({"role": "assistant", "content": ChatGPT_reply})
#     return ChatGPT_reply

# demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title = "Real Estate Pro")

# demo.launch(share=True)

#step 4 web app using streamlit

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a financial expert that specializes in real estate investment and negotiation."}
    ]

st.title("🏡 Real Estate AI Pro")

for message in st.session_state.messages:
    if message["role"] != "system":  # Don't show system messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_input = st.chat_input("Ask about real estate investment...")

if user_input:
    # Append user input to message history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response from OpenAI
    response = openai.ChatCompletion.create(
        engine=AZURE_DEPLOYMENT_NAME,  # Use GPT-4 or GPT-4o if needed
        messages=st.session_state.messages
    )

    # Get GPT's reply
    assistant_reply = response["choices"][0]["message"]["content"]

    # Append assistant response to message history
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

    # Display assistant response in chat
    with st.chat_message("assistant"):
        st.write(assistant_reply)