"""
Example code with intentional issues for testing the AI code reviewer.
This demonstrates various types of problems the reviewer can catch.
"""

import sqlite3
import hashlib
from fastapi import FastAPI, Request

app = FastAPI()

# Security Issue 1: Hardcoded secret
SECRET_KEY = "my-super-secret-key-123"

# Security Issue 2: SQL injection vulnerability
def get_user_by_name(username):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    # Vulnerable SQL query
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

# Performance Issue 1: Inefficient algorithm
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

# Performance Issue 2: No caching for expensive operation
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Style Issue 1: Poor naming and no documentation
def func(x, y):
    return x + y

# Style Issue 2: No error handling
@app.post("/login")
def login(request: Request):
    data = request.json()
    username = data["username"]  # Could throw KeyError
    password = data["password"]  # Could throw KeyError
    
    # Security Issue 3: Weak password hashing
    hashed = hashlib.md5(password.encode()).hexdigest()
    
    user = get_user_by_name(username)
    if user and user[2] == hashed:
        return {"success": True, "user_id": user[0]}
    return {"success": False}

# Performance Issue 3: Loading large data repeatedly
def process_user_data(user_id):
    # This would load all users every time
    all_users = load_all_users_from_database()
    user = None
    for u in all_users:
        if u["id"] == user_id:
            user = u
            break
    
    if user:
        return calculate_user_score(user)
    return None

def load_all_users_from_database():
    # Simulates loading large dataset
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users")
    results = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "password": r[2]} for r in results]

# Style Issue 3: Long function with multiple responsibilities
def calculate_user_score(user):
    score = 0
    
    # Calculate based on activity
    if user.get("last_login"):
        days_since_login = (datetime.now() - user["last_login"]).days
        if days_since_login < 7:
            score += 10
        elif days_since_login < 30:
            score += 5
    
    # Calculate based on posts
    posts = get_user_posts(user["id"])
    score += len(posts) * 2
    
    # Calculate based on friends
    friends = get_user_friends(user["id"])
    score += len(friends)
    
    # Send notification email
    send_score_email(user["email"], score)
    
    # Log to analytics
    log_user_score(user["id"], score)
    
    return score

# Security Issue 4: No input validation
@app.post("/update_profile")
def update_profile(request: Request):
    data = request.json()
    user_id = data["user_id"]
    new_email = data["email"]  # No email validation
    new_bio = data["bio"]      # No length limits
    
    # Direct database update without sanitization
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    query = f"UPDATE users SET email = '{new_email}', bio = '{new_bio}' WHERE id = {user_id}"
    cursor.execute(query)
    conn.commit()
    conn.close()
    
    return {"success": True}
