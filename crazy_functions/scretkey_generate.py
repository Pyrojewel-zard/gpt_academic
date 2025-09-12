import secrets
import string

def generate_sk_key(length: int = 48) -> str:
    alphabet = string.ascii_letters + string.digits
    body = ''.join(secrets.choice(alphabet) for _ in range(length))
    return f"sk-{body}"

if __name__ == "__main__":
    print(generate_sk_key())