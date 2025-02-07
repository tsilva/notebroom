from openai import OpenAI

SYSTEM_PROMPT = """
Rewrite markdown text to be more concise and clear. Preserve meaning and formatting. Return only the revised markdown.
""".strip()

def rewrite_text(text):
    """Rewrite text using LLM"""
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return resp.choices[0].message.content.strip()
