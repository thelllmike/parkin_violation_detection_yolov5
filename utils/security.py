# utils/security.py
import os
from dotenv import load_dotenv
load_dotenv()            # ← ensure .env is loaded here
from cryptography.fernet import Fernet, InvalidToken

FERNET_KEY = os.getenv("FERNET_KEY")
if not FERNET_KEY:
    raise RuntimeError("Set FERNET_KEY in your environment")

fernet = Fernet(FERNET_KEY.encode())

def encrypt_key(raw_private_key: str) -> str:
    """
    Encrypts a raw private key (hex string) into a token you can safely store in DB.
    """
    token = fernet.encrypt(raw_private_key.encode())
    return token.decode()


def decrypt_key(token: str) -> str:
    """
    Decrypts the token back into the original private key.
    Raises InvalidToken if tampering or wrong key.
    """
    try:
        raw = fernet.decrypt(token.encode())
        return raw.decode()
    except InvalidToken:
        raise ValueError("Invalid encryption token for private key")