"""
Login and roles for StudentPulse AI.

Roles
  staff       - faculty / mentors: dashboard, student explorer, early warning, dataset (view), own account
  management  - HOD / principal / admin: everything staff can do, plus management summary,
                model report + retraining, data upload, user management and the access log

How it works (easy to explain in a viva)
  * Passwords are never stored as text. Each password is salted and hashed with PBKDF2-SHA256.
  * Accounts live in data/users.json (hashes only). Default demo accounts are created on first run.
  * Every login, failed login and logout is written to data/access_log.csv (audit trail).
  * After 5 wrong attempts the login is locked for 60 seconds.

This is suitable for a college demo / internal pilot. For a real deployment, use the college
single sign-on (SSO) instead of a local user file.
"""
import csv
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
USERS_FILE = BASE / "data" / "users.json"
LOG_FILE = BASE / "data" / "access_log.csv"

ROLES = ("staff", "management")
ROLE_LABEL = {"staff": "Staff", "management": "Management"}
ITERATIONS = 200_000
MAX_ATTEMPTS = 5
LOCK_SECONDS = 60

# Demo accounts created on the first run. Change these passwords (Management > User Management).
DEFAULT_ACCOUNTS = [
    {"username": "staff", "name": "Faculty Mentor", "role": "staff", "password": "Staff@123"},
    {"username": "hod", "name": "Head of Department", "role": "management", "password": "Hod@123"},
    {"username": "principal", "name": "Principal", "role": "management", "password": "Principal@123"},
]


# ------------------------------------------------------------------ hashing
def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), ITERATIONS)
    return salt, digest.hex()


def _check(password, salt, stored_hash):
    _, digest = hash_password(password, salt)
    return hmac.compare_digest(digest, stored_hash)


# ------------------------------------------------------------------ user store
def _load():
    if not USERS_FILE.exists():
        users = {}
        for acc in DEFAULT_ACCOUNTS:
            salt, h = hash_password(acc["password"])
            users[acc["username"]] = {"name": acc["name"], "role": acc["role"], "salt": salt, "hash": h}
        _save(users)
        return users
    return json.loads(USERS_FILE.read_text())


def _save(users):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, indent=2))


def verify_login(username, password, role):
    """Return {'username','name','role'} if the credentials and the chosen role are correct, else None."""
    users = _load()
    u = users.get(username.strip().lower())
    # always do the hash work so timing does not reveal whether a username exists
    salt = u["salt"] if u else "00" * 16
    ok = _check(password, salt, u["hash"] if u else "0" * 64)
    if u and ok and u["role"] == role:
        return {"username": username.strip().lower(), "name": u["name"], "role": u["role"]}
    return None


def list_users():
    return [{"username": k, "name": v["name"], "role": v["role"]} for k, v in _load().items()]


def password_problem(password):
    """Return a message if the password is too weak, else None."""
    if len(password) < 8:
        return "Use at least 8 characters."
    if password.lower() == password or not any(c.isdigit() for c in password):
        return "Use upper and lower case letters and at least one digit."
    return None


def add_user(username, name, role, password):
    username = username.strip().lower()
    if not username.isalnum():
        return "Username may contain letters and numbers only."
    if role not in ROLES:
        return "Invalid role."
    if not name.strip():
        return "Please enter the person's name."
    problem = password_problem(password)
    if problem:
        return problem
    users = _load()
    if username in users:
        return "That username already exists."
    salt, h = hash_password(password)
    users[username] = {"name": name.strip(), "role": role, "salt": salt, "hash": h}
    _save(users)
    return None


def change_password(username, old_password, new_password):
    users = _load()
    u = users.get(username)
    if not u or not _check(old_password, u["salt"], u["hash"]):
        return "Current password is incorrect."
    problem = password_problem(new_password)
    if problem:
        return problem
    u["salt"], u["hash"] = hash_password(new_password)
    _save(users)
    return None


def reset_password(username, new_password):
    """Management action: set a new password for someone who forgot theirs."""
    users = _load()
    if username not in users:
        return "No such user."
    problem = password_problem(new_password)
    if problem:
        return problem
    users[username]["salt"], users[username]["hash"] = hash_password(new_password)
    _save(users)
    return None


def remove_user(username, acting_username):
    users = _load()
    if username == acting_username:
        return "You cannot remove your own account."
    if username not in users:
        return "No such user."
    if users[username]["role"] == "management" and sum(v["role"] == "management" for v in users.values()) <= 1:
        return "At least one management account must remain."
    del users[username]
    _save(users)
    return None


# ------------------------------------------------------------------ lockout (per browser session)
def lock_seconds_left(state):
    """`state` is st.session_state (any dict-like). Returns seconds of lockout remaining (0 = not locked)."""
    until = state.get("lock_until", 0)
    return max(0, int(until - time.time()))


def register_failure(state):
    state["failed"] = state.get("failed", 0) + 1
    if state["failed"] >= MAX_ATTEMPTS:
        state["lock_until"] = time.time() + LOCK_SECONDS
        state["failed"] = 0


def register_success(state):
    state["failed"] = 0
    state["lock_until"] = 0


# ------------------------------------------------------------------ audit log
def log_event(username, role, event):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "username", "role", "event"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username, role, event])


def read_log():
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, newline="") as f:
        return list(csv.DictReader(f))[::-1]  # newest first
