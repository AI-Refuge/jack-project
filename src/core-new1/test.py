from together import Together

client = Together()

response = client.chat.completions.create(
    model="kuldeepdhaka/Meta-Llama-3-70B-Instruct-meta-002-0a4c3de5-8d151bf8",
    messages=[
        {"role": "user", "content": open("meta.txt").read()}
        {"role": "user", "content": "hi!"}
    ],
)
print(response.choices[0].message.content)
