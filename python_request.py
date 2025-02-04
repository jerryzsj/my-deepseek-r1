import openai
client = openai.Client(
    base_url="http://127.0.0.1:30000/v1", api_key='None')

# Chat completion
response = client.chat.completions.create(
    model="/root/.cache/DeepSeek-R1-Distill-Qwen-7B",
    messages=[
        {"role": "user", "content": "总结一下A股2024的表现"},
        {"role": "user", "content": "给我一个2025的投资建议"},
    ],
    temperature=0.7,
    max_tokens=50000,
)
print(response)
