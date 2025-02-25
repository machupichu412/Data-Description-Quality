import ollama

response = ollama.chat(model='vicuna', messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ])

print(response['message']['content'])