from Crypto.Cipher import AES
import base64
import hashlib

class AESCipher:
    def __init__(self, key="SecureKey123456!"): 
        """Initialize AES encryption with a 256-bit hashed key."""
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, data):
        """Encrypts data using AES."""
        cipher = AES.new(self.key, AES.MODE_EAX)
        nonce = cipher.nonce
        ciphertext, tag = cipher.encrypt_and_digest(data.encode())
        return base64.b64encode(nonce + ciphertext).decode()

    def decrypt(self, encrypted_data):
        """Decrypts AES-encrypted data."""
        raw_data = base64.b64decode(encrypted_data)
        nonce, ciphertext = raw_data[:16], raw_data[16:]
        cipher = AES.new(self.key, AES.MODE_EAX, nonce=nonce)
        return cipher.decrypt(ciphertext).decode()

# from Crypto.Cipher import AES
# import base64
# import hashlib

# class AESCipher:
#     def __init__(self, key="SecureKey123456!"):
#         """Initialize AES encryption with a 256-bit hashed key."""
#         self.key = hashlib.sha256(key.encode()).digest()

#     def encrypt(self, data):
#         """Encrypts data using AES."""
#         if not isinstance(data, str):
#             raise ValueError("Data to encrypt must be a string")

#         cipher = AES.new(self.key, AES.MODE_EAX)
#         nonce = cipher.nonce
#         ciphertext, tag = cipher.encrypt_and_digest(data.encode())

#         # Store nonce + ciphertext together for decryption
#         encrypted_data = base64.b64encode(nonce + ciphertext).decode()
#         return encrypted_data

#     def decrypt(self, encrypted_data):
#         """Decrypts AES-encrypted data."""
#         try:
#             raw_data = base64.b64decode(encrypted_data)
#             nonce, ciphertext = raw_data[:16], raw_data[16:]

#             cipher = AES.new(self.key, AES.MODE_EAX, nonce=nonce)
#             decrypted_data = cipher.decrypt(ciphertext).decode()
#             return decrypted_data

#         except (ValueError, TypeError, base64.binascii.Error) as e:
#             print(f"Decryption error: {e}")
#             return None  # Return None if decryption fails

# # Example usage
# if __name__ == "__main__":
#     aes = AESCipher()

#     # Encrypt
#     original_text = "Hello, this is a secret message!"
#     encrypted_text = aes.encrypt(original_text)
#     print("Encrypted:", encrypted_text)

#     # Decrypt
#     decrypted_text = aes.decrypt(encrypted_text)
#     print("Decrypted:", decrypted_text)
#     assert decrypted_text == original_text, "Decryption failed!"
    