
import sqlite3
import os

DATABASE_PATH = os.path.join("auth", "users.db")


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_users_table():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def create_user(name, email, hashed_password):
    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO users (name, email, hashed_password)
            VALUES (?, ?, ?)
            """,
            (name, email, hashed_password)
        )

        connection.commit()
        return cursor.lastrowid

    except sqlite3.IntegrityError:
        return None

    finally:
        connection.close()


def get_user_by_email(email):
    connection = get_connection()

    user = connection.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    connection.close()
    return user
