import os

from openai import OpenAI  # pyright: ignore[reportMissingImports]


gateway_api_key = os.environ.get("GW_API_KEY")
if not gateway_api_key or not gateway_api_key.startswith("gw_"):
    raise RuntimeError("Set GW_API_KEY to your generated gw_... gateway key.")

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=gateway_api_key,
)
r = client.chat.completions.create(
    model="gemini-flash",
    messages=[{"role": "user", "content": "hello gateway"}],
)
print(r.choices[0].message.content, r.usage)
