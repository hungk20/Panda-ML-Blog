First, register OpenAI account & get your API key at [OpenAI API keys](https://platform.openai.com/account/api-keys) and replace it in the line

```
openai.api_key = 'YOUR_OPEN_API_KEY'
```

After that, you can just start the Chatbot by the following command
```
streamlit run chat_with_your_data.py
```

Required libraries:
```
streamlit
streamlit-chat
langchain
faiss-cpu
```
