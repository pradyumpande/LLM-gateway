from openai import OpenAI  # pyright: ignore[reportMissingImports]


client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything-for-now")
r = client.chat.completions.create(
    model="gemini-flash",
    messages=[{"role": "user", "content": "hello gateway"}],
)
print(r.choices[0].message.content, r.usage)
