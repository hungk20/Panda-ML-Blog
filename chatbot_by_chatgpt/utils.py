import openai


openai.api_key = 'YOUR_OPEN_API_KEY'


def get_completion_from_messages(messages, model='gpt-3.5-turbo', temperatute=0):
    """ Send list of messages to OpenAI and get the response 
    
    Example
      - Input
        [
            {'role': 'system', 'content': 'you are an assistant, answering in a concise, friendly manner'},
            {'role': 'user', 'content': 'Hi'},
        ]
      - Response
        'Hey, how are you today?'

    Parameters:
    -----------
    messages: List[Dict]
        List of dictionary that store the chat messages role & content
    model: Str
        String represent the model to be used
    temperature: Float
        Randomness of the answer (the higher the value, the more random it will be)
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperatute,
    )
    return response.choices[0].message['content']


def convert_st_chat_to_openai_format(history):
    """ Helper function to convert streamlit messages format into OpenAI format 
    
    streamlit-chat format                             => OpenAI format
    {'message': 'Hi', is_user: True}                  => {'role': 'user', 'content': 'Hi'}
    {'message': 'How are you today?', is_user: False} => {'role': 'assistant', 'content': 'How are you today?'}
    
    """
    messages = []
    for chat in history:
        role = 'user' if chat['is_user'] else 'assistant'
        prompt = chat['message']
        messages.append({'role': role, 'content': f'{prompt}'})

    return messages
