# import sqlite3
# import hashlib
# from aes_crypto import AESCipher

# class Database:
#     def __init__(self, db_name="users.db"):
#         """Initialize database and create table if not exists."""
#         self.conn = sqlite3.connect(db_name)
#         self.cursor = self.conn.cursor()
#         self.aes = AESCipher()
#         self.create_table()

#     def create_table(self):
#         """Create users table."""
#         self.cursor.execute('''CREATE TABLE IF NOT EXISTS users (
#                                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                 username TEXT UNIQUE,
#                                 password TEXT,
#                                 click_points TEXT,
#                                 image_sequence TEXT)''')
#         self.conn.commit()

#     def hash_password(self, password):
#         """Hashes the password using SHA-256."""
#         return hashlib.sha256(password.encode()).hexdigest()

#     def register_user(self, username, password, click_points, image_sequence):
#         """Register a user with hashed password and encrypted click points."""
#         hashed_password = self.hash_password(password)
#         encrypted_click_points = self.aes.encrypt(str(click_points))
#         encrypted_image_sequence = self.aes.encrypt(str(image_sequence))

#         try:
#             self.cursor.execute("INSERT INTO users (username, password, click_points, image_sequence) VALUES (?, ?, ?, ?)",
#                                 (username, hashed_password, encrypted_click_points, encrypted_image_sequence))
#             self.conn.commit()
#             return True
#         except sqlite3.IntegrityError:
#             print("Username already exists!")
#             return False

#     def authenticate_user(self, username, entered_password, entered_click_points, entered_image_sequence):
#         """Authenticate a user by verifying credentials."""
#         self.cursor.execute("SELECT password, click_points, image_sequence FROM users WHERE username=?", (username,))
#         result = self.cursor.fetchone()

#         if not result:
#             print("User not found!")
#             return False

#         stored_password, stored_encrypted_click_points, stored_encrypted_image_sequence = result

#         # Verify password
#         if self.hash_password(entered_password) != stored_password:
#             print("Incorrect password!")
#             return False

#         # Decrypt stored click points & image sequence
#         stored_click_points = eval(self.aes.decrypt(stored_encrypted_click_points))
#         stored_image_sequence = eval(self.aes.decrypt(stored_encrypted_image_sequence))

#         # Check image sequences match
#         if entered_image_sequence != stored_image_sequence:
#             print("Image sequences do not match!")
#             return False

#         # Tolerance check for click points
#         tolerance = 10  # Allow 10-pixel margin
#         for stored, entered in zip(stored_click_points, entered_click_points):
#             for (sx, sy), (ex, ey) in zip(stored, entered):
#                 if abs(sx - ex) > tolerance or abs(sy - ey) > tolerance:
#                     print("Click points do not match within tolerance!")
#                     return False

#         print("Login Successful!")
#         return True

# # import sqlite3
# # import hashlib
# # from aes_crypto import AESCipher  # Ensure this module exists

# # class Database:
# #     def __init__(self, db_name="users.db"):
# #         """Initialize database and create users table if it doesn't exist."""
# #         self.conn = sqlite3.connect(db_name)
# #         self.cursor = self.conn.cursor()
# #         self.aes = AESCipher()
# #         self.create_table()

# #     def create_table(self):
# #         """Create users table in the database."""
# #         self.cursor.execute('''
# #             CREATE TABLE IF NOT EXISTS users (
# #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
# #                 username TEXT UNIQUE NOT NULL,
# #                 password TEXT NOT NULL,
# #                 click_points TEXT NOT NULL,
# #                 image_sequence TEXT NOT NULL
# #             )
# #         ''')
# #         self.conn.commit()

# #     def hash_password(self, password):
# #         """Hashes the password using SHA-256."""
# #         return hashlib.sha256(password.encode()).hexdigest()

# #     def register_user(self, username, password, click_points, image_sequence):
# #         """Register a new user with hashed password and encrypted graphical password."""
# #         hashed_password = self.hash_password(password)
# #         encrypted_click_points = self.aes.encrypt(str(click_points))
# #         encrypted_image_sequence = self.aes.encrypt(str(image_sequence))

# #         try:
# #             self.cursor.execute("""
# #                 INSERT INTO users (username, password, click_points, image_sequence)
# #                 VALUES (?, ?, ?, ?)
# #             """, (username, hashed_password, encrypted_click_points, encrypted_image_sequence))
# #             self.conn.commit()
# #             print("User registered successfully!")
# #             return True
# #         except sqlite3.IntegrityError:
# #             print("Error: Username already exists!")
# #             return False

# #     def authenticate_user(self, username, entered_password, entered_click_points, entered_image_sequence):
# #         """Authenticate a user by verifying credentials and click points."""
# #         self.cursor.execute("SELECT password, click_points, image_sequence FROM users WHERE username=?", (username,))
# #         result = self.cursor.fetchone()

# #         if not result:
# #             print("Error: User not found!")
# #             return False

# #         stored_password, stored_encrypted_click_points, stored_encrypted_image_sequence = result

# #         # Verify hashed password
# #         if self.hash_password(entered_password) != stored_password:
# #             print("Error: Incorrect password!")
# #             return False

# #         # Decrypt stored click points and image sequence
# #         stored_click_points = eval(self.aes.decrypt(stored_encrypted_click_points))
# #         stored_image_sequence = eval(self.aes.decrypt(stored_encrypted_image_sequence))

# #         # Check if image sequences match
# #         if entered_image_sequence != stored_image_sequence:
# #             print("Error: Image sequences do not match!")
# #             return False

# #         # Tolerance check for click points
# #         tolerance = 10  # Allow 10-pixel margin
# #         for stored, entered in zip(stored_click_points, entered_click_points):
# #             for (sx, sy), (ex, ey) in zip(stored, entered):
# #                 if abs(sx - ex) > tolerance or abs(sy - ey) > tolerance:
# #                     print("Error: Click points do not match within tolerance!")
# #                     return False

# #         print("Login Successful!")
# #         return True

# # # Example usage
# # if __name__ == "__main__":
# #     db = Database()

# #     # Register a user
# #     db.register_user(
# #         username="test_user",
# #         password="SecurePass123",
# #         click_points=[[(100, 200), (250, 300)]],  # Example points
# #         image_sequence=["image1.jpg", "image2.jpg"]
# #     )

# #     # Authenticate user
# #     success = db.authenticate_user(
# #         username="test_user",
# #         entered_password="SecurePass123",
# #         entered_click_points=[[(100, 200), (250, 300)]],  # Same points
# #         entered_image_sequence=["image1.jpg", "image2.jpg"]
# #     )

# #     print("Authentication Success:", success)


import sqlite3
import json
import os
from argon2 import PasswordHasher
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

class AESCipher:
    def __init__(self, key=None):
        """Initialize AES with a key (if not provided, generate a new one)."""
        self.key = key if key else os.urandom(32)  # 256-bit key
        self.iv = os.urandom(16)  # 128-bit IV (Initialization Vector)

    def encrypt(self, plaintext):
        """Encrypt plaintext using AES CBC mode."""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode()
        
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Combine IV and ciphertext and encode as base64
        return base64.b64encode(self.iv + ciphertext).decode('utf-8')

    def decrypt(self, encrypted_text):
        """Decrypt ciphertext using AES CBC mode."""
        # Decode from base64 and extract IV
        encrypted_data = base64.b64decode(encrypted_text.encode('utf-8'))
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()

        return data.decode('utf-8')

class Database:
    def __init__(self):
        self.db_file = 'users.db'
        self.aes = AESCipher()
        self.ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,  # 64MB
            parallelism=4,
            hash_len=32,
            salt_len=16
        )
        self.create_table()

    def create_table(self):
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                # Drop existing table if it exists
                cursor.execute('''DROP TABLE IF EXISTS users''')
                cursor.execute('''DROP TABLE IF EXISTS model_metrics''')
                
                # Create users table with Argon2id hash storage
                cursor.execute('''
                    CREATE TABLE users (
                        username TEXT PRIMARY KEY,
                        security_question TEXT NOT NULL,
                        security_answer_hash TEXT NOT NULL,
                        click_points TEXT NOT NULL,
            image_sequence TEXT NOT NULL,
                        salt TEXT NOT NULL
                    )
                ''')
                
                # Create model metrics table
                cursor.execute('''
                    CREATE TABLE model_metrics (
                        username TEXT PRIMARY KEY,
                        resnet_accuracy REAL,
                        resnet_precision REAL,
                        resnet_recall REAL,
                        vit_accuracy REAL,
                        vit_precision REAL,
                        vit_recall REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (username) REFERENCES users(username)
                    )
                ''')
                conn.commit()
        except Exception as e:
            print(f"Error creating table: {str(e)}")

    def register_user(self, username, security_question, security_answer, click_points, image_sequence):
        try:
            # Generate a unique salt for the user
            salt = os.urandom(16).hex()
            
            # Hash the security answer with Argon2id
            security_answer_hash = self.ph.hash(security_answer + salt)
            
            # Convert click points and image sequence to JSON strings
            click_points_json = json.dumps(click_points)
            image_sequence_json = json.dumps(image_sequence)
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (
                        username, security_question, security_answer_hash,
                        click_points, image_sequence, salt
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    username,
                    security_question,
                    security_answer_hash,
                    click_points_json,
                    image_sequence_json,
                    salt
                ))
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            print(f"Error registering user: {str(e)}")
            return False

    def update_model_metrics(self, username, metrics):
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO model_metrics (
                        username, 
                        resnet_accuracy, resnet_precision, resnet_recall,
                        vit_accuracy, vit_precision, vit_recall
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    username,
                    metrics['resnet_metrics']['accuracy'],
                    metrics['resnet_metrics']['precision'],
                    metrics['resnet_metrics']['recall'],
                    metrics['vit_metrics']['accuracy'],
                    metrics['vit_metrics']['precision'],
                    metrics['vit_metrics']['recall']
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error updating model metrics: {str(e)}")
            return False

    def get_model_metrics(self, username):
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        resnet_accuracy, resnet_precision, resnet_recall,
                        vit_accuracy, vit_precision, vit_recall,
                        timestamp
                    FROM model_metrics 
                    WHERE username = ?
                ''', (username,))
                result = cursor.fetchone()
                
                if result:
                    return {
                        'resnet_metrics': {
                            'accuracy': result[0],
                            'precision': result[1],
                            'recall': result[2]
                        },
                        'vit_metrics': {
                            'accuracy': result[3],
                            'precision': result[4],
                            'recall': result[5]
                        },
                        'timestamp': result[6]
                    }
                return None
        except Exception as e:
            print(f"Error retrieving model metrics: {str(e)}")
            return None

    def get_user_data(self, username):
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT security_question, click_points, image_sequence 
                    FROM users 
                    WHERE username = ?
                ''', (username,))
                result = cursor.fetchone()
                
                if result:
                    # Parse JSON strings back to Python objects
                    try:
                        return {
                            'security_question': result[0],
                            'click_points': json.loads(result[1]),
                            'image_sequence': json.loads(result[2])
                        }
                    except json.JSONDecodeError as e:
                        raise Exception(f"Invalid data format in database: {str(e)}")
                return None
        except Exception as e:
            raise Exception(f"Error retrieving user data: {str(e)}")

    def authenticate_user(self, username, security_answer, provided_click_points, provided_image_sequence):
        """Authenticate a user by verifying their security answer and click points."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT security_answer_hash, click_points, image_sequence, salt
                    FROM users WHERE username = ?
                ''', (username,))
                result = cursor.fetchone()
                
                if not result:
                    return False
                
                stored_hash, stored_click_points, stored_image_sequence, salt = result
                
                # Verify security answer using Argon2id
                try:
                    self.ph.verify(stored_hash, security_answer + salt)
                except:
                    return False
                
                # Parse stored click points and image sequence
                stored_click_points = json.loads(stored_click_points)
                stored_image_sequence = json.loads(stored_image_sequence)
                
                # Check image sequences match
                if stored_image_sequence != provided_image_sequence:
                    return False
                
                # Point comparison with tolerance
                for stored_points, provided_points in zip(stored_click_points, provided_click_points):
                    if len(stored_points) != len(provided_points):
                        return False
                    for (sx, sy), (px, py) in zip(stored_points, provided_points):
                        # Allow for some tolerance in click positions (within 20 pixels)
                        if abs(sx - px) > 20 or abs(sy - py) > 20:
                            return False

                return True
                
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False

    def close_connection(self):
        """Close the database connection."""
        self.conn.close()
