import os
import time
from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError, APITimeoutError, APIConnectionError, BadRequestError, APIStatusError

load_dotenv()
MODEL_NAME = "openai/gpt-oss-120b"   

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def call_llm(messages, max_retries=3, timeout_seconds=30):
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                timeout=timeout_seconds,
            )
            return response.choices[0].message.content, None

        except RateLimitError:
            wait_time = 2 ** attempt  
            print(f"Rate limit hit. Retry {attempt}/{max_retries}, waiting {wait_time}s...")
            time.sleep(wait_time)

        except (APITimeoutError, APIConnectionError) as e:
            print(f"Connection/timeout issue. Retry {attempt}/{max_retries}: {e}")
            time.sleep(2)

        except BadRequestError as e:
            return None, f"Request error (the prompt may be too long): {e}"

        except APIStatusError as e:
            return None, f"The API returned an error: {e}"

    return None, f"No response after {max_retries} attempts. Please try again later."


if __name__ == "__main__":
    messages = [{"role": "user", "content": "Say hello in one short sentence."}]
    answer, error = call_llm(messages)

    if error:
        print("ERROR:", error)
    else:
        print("ANSWER:", answer)