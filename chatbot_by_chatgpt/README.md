First, register OpenAI account & get your API key at [OpenAI API keys](https://platform.openai.com/account/api-keys) and replace it in the line

```
openai.api_key = 'YOUR_OPEN_API_KEY'
```

After that, you can just start the Chatbot by the following command
```
streamlit run order_chatbot.py
```

Note that you need to install 3 libraries: `streamlit`, `streamlit-chat`, `openai` before running this script.