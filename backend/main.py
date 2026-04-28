"""
Backend API for Promjai-Klaibaan MVP
====================================
A FastAPI application providing:
  - User auth (hashed password, token with TTL)
  - Activity recommendations (Bi-Level GAT-powered)
  - Caretaker location-based matching (seed + registered)
  - WHOQOL-BREF tracking
  - Activity logging + ratings
  - Research-grade event logging (JSONL)

Loaded model: data/embeddings.npz (precomputed at training time).

Run:
  python backend/main.py
or
  uvicorn backend.main:app --reload --port 8000

ID encoding in bookings table
-------------------------------
  elderly_id   >= 0  → index into SEED_USERS (CSV)
  elderly_id    < 0  → registered user, real user_id = -elderly_id
  caretaker_id >= 0  → index into SEED_CARETAKERS (CSV)
  caretaker_id  < 0  → registered user, real user_id = -caretaker_id
"""
import os
import csv
import json
import math
import time
import hmac
import hashlib
import base64
import sqlite3
import secrets
import logging
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger('promjai')

import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
ML   = ROOT / 'ml'
# Allow env-override so cloud deploys can point to a persistent-disk path
LOG_DIR = Path(os.environ.get('LOG_DIR', str(ROOT / 'logs')))
LOG_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.environ.get('DB_PATH', str(ROOT / 'promjai.db')))

TOKEN_TTL_HOURS = int(os.environ.get('TOKEN_TTL_HOURS', '24'))
ALLOWED_ORIGINS = os.environ.get(
    'ALLOWED_ORIGINS', 'http://localhost:8000'
).split(',')

# ---------- Thai province → approximate centre coordinates ----------
PROVINCE_COORDS: dict[str, tuple[float, float]] = {
    'กรุงเทพมหานคร': (13.7563, 100.5018),
    'กระบี่':        (8.0863,   98.9063),
    'กาญจนบุรี':     (14.0039,  99.5476),
    'กาฬสินธุ์':     (16.4322, 103.5062),
    'กำแพงเพชร':    (16.4827,  99.5226),
    'ขอนแก่น':       (16.4419, 102.8360),
    'จันทบุรี':      (12.6113, 102.1035),
    'ฉะเชิงเทรา':   (13.6904, 101.0779),
    'ชลบุรี':        (13.3611, 100.9851),
    'ชัยนาท':       (15.1850, 100.1257),
    'ชัยภูมิ':      (15.8068, 102.0319),
    'ชุมพร':         (10.4930,  99.1800),
    'เชียงราย':     (19.9105,  99.8406),
    'เชียงใหม่':    (18.7883,  98.9853),
    'ตรัง':          (7.5593,  99.6112),
    'ตราด':          (12.2437, 102.5162),
    'ตาก':           (16.8798,  99.1265),
    'นครนายก':      (14.2069, 101.2132),
    'นครปฐม':       (13.8199, 100.0624),
    'นครพนม':       (17.3922, 104.7693),
    'นครราชสีมา':   (14.9799, 102.0978),
    'นครศรีธรรมราช': (8.4324,  99.9633),
    'นครสวรรค์':    (15.7030, 100.1368),
    'นนทบุรี':       (13.8621, 100.5144),
    'นราธิวาส':     (6.4254,  101.8253),
    'น่าน':          (18.7833, 100.7794),
    'บึงกาฬ':        (18.3609, 103.6461),
    'บุรีรัมย์':    (14.9934, 103.1029),
    'ปทุมธานี':     (14.0208, 100.5250),
    'ประจวบคีรีขันธ์': (11.8126, 99.7957),
    'ปราจีนบุรี':   (14.0509, 101.3733),
    'ปัตตานี':       (6.8698,  101.2502),
    'พระนครศรีอยุธยา': (14.3692, 100.5877),
    'พะเยา':         (19.1663,  99.9009),
    'พังงา':         (8.4508,   98.5258),
    'พัทลุง':        (7.6166,  100.0742),
    'พิจิตร':        (16.4419, 100.3491),
    'พิษณุโลก':     (16.8211, 100.2659),
    'เพชรบุรี':     (13.1119,  99.9392),
    'เพชรบูรณ์':    (16.4189, 101.1604),
    'แพร่':          (18.1445, 100.1402),
    'ภูเก็ต':        (7.8804,   98.3923),
    'มหาสารคาม':    (16.1851, 103.3009),
    'มุกดาหาร':     (16.5436, 104.7237),
    'แม่ฮ่องสอน':  (19.3020,  97.9654),
    'ยโสธร':         (15.7921, 104.1455),
    'ยะลา':           (6.5414, 101.2804),
    'ร้อยเอ็ด':     (16.0538, 103.6520),
    'ระนอง':          (9.9528,  98.6084),
    'ระยอง':         (12.6814, 101.2816),
    'ราชบุรี':       (13.5360,  99.8172),
    'ลพบุรี':        (14.7995, 100.6534),
    'ลำปาง':         (18.2888,  99.4925),
    'ลำพูน':         (18.5744,  99.0087),
    'เลย':           (17.4860, 101.7224),
    'ศรีสะเกษ':     (15.1186, 104.3221),
    'สกลนคร':       (17.1551, 104.1348),
    'สงขลา':         (7.1756,  100.6142),
    'สตูล':           (6.6238, 100.0678),
    'สมุทรปราการ':  (13.5991, 100.5998),
    'สมุทรสงคราม':  (13.4098, 100.0022),
    'สมุทรสาคร':    (13.5474, 100.2744),
    'สระแก้ว':       (13.8240, 102.0648),
    'สระบุรี':       (14.5289, 100.9101),
    'สิงห์บุรี':    (14.8936, 100.3968),
    'สุโขทัย':       (17.0068,  99.8265),
    'สุพรรณบุรี':   (14.4744, 100.1177),
    'สุราษฎร์ธานี': (9.1382,   99.3222),
    'สุรินทร์':      (14.8827, 103.4938),
    'หนองคาย':       (17.8782, 102.7421),
    'หนองบัวลำภู':  (17.2218, 102.4260),
    'อ่างทอง':       (14.5896, 100.4551),
    'อำนาจเจริญ':   (15.8656, 104.6257),
    'อุดรธานี':     (17.4138, 102.7876),
    'อุตรดิตถ์':    (17.6200, 100.0993),
    'อุทัยธานี':    (15.3835, 100.0255),
    'อุบลราชธานี':  (15.2287, 104.8597),
}

# ---------- Rate limiting (in-memory) ----------
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_RATE_LIMIT_WINDOW  = float(os.environ.get('RATE_WINDOW', '60'))
_RATE_LIMIT_MAX     = int(os.environ.get('RATE_MAX', '10'))


def _check_rate_limit(key: str) -> None:
    now = time.time()
    recent = [t for t in _LOGIN_ATTEMPTS.get(key, []) if now - t < _RATE_LIMIT_WINDOW]
    if len(recent) >= _RATE_LIMIT_MAX:
        raise HTTPException(429, 'Too many login attempts. Try again later.')
    recent.append(now)
    _LOGIN_ATTEMPTS[key] = recent


# ---------- Password hashing (PBKDF2-HMAC-SHA256, 260k rounds) ----------
_PW_PREFIX = '$pbkdf2$'


def hash_pw(password: str) -> str:
    """Hash with PBKDF2-HMAC-SHA256 + random 16-byte salt."""
    salt = os.urandom(16)
    key  = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 260_000)
    return _PW_PREFIX + base64.b64encode(salt + key).decode()


def _verify_pw(password: str, stored: str) -> bool:
    """Constant-time verify; handles legacy SHA-256 during migration."""
    if stored.startswith(_PW_PREFIX):
        try:
            raw        = base64.b64decode(stored[len(_PW_PREFIX):])
            salt, key  = raw[:16], raw[16:]
            chk        = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 260_000)
            return hmac.compare_digest(key, chk)
        except Exception:
            return False
    # Legacy SHA-256 (64 hex chars) — allow once, init_db will upgrade on restart
    if len(stored) == 64:
        return hmac.compare_digest(
            stored.encode(),
            hashlib.sha256(password.encode()).hexdigest().encode()
        )
    return False


# ---------- DB ----------
@contextmanager
def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with db_conn() as conn:
        c = conn.cursor()

        # 1. Create core tables (idempotent)
        c.executescript('''
        CREATE TABLE IF NOT EXISTS app_users (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            username         TEXT UNIQUE NOT NULL,
            password         TEXT NOT NULL,
            role             TEXT NOT NULL,       -- "elderly" | "caretaker"
            elderly_id       INTEGER,             -- FK to seed users.csv (seed users only)
            caretaker_id     INTEGER,             -- FK to caretakers.csv (seed caretakers only)
            token            TEXT,
            token_created_at TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS activity_logs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            activity_id  INTEGER NOT NULL,
            rating       REAL,
            qol_delta    REAL,
            completed_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS qol_history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            physical      REAL,
            psychological REAL,
            social        REAL,
            environment   REAL,
            recorded_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bookings (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            elderly_id   INTEGER NOT NULL,
            caretaker_id INTEGER NOT NULL,
            service_type TEXT,
            scheduled_at TEXT,
            status       TEXT DEFAULT "pending",
            rating       REAL,
            feedback     TEXT,
            created_at   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            event_type TEXT,
            payload    TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id      INTEGER PRIMARY KEY,
            name         TEXT,
            birth_date   TEXT,
            gender       TEXT,
            province     TEXT,
            district     TEXT,
            subdistrict  TEXT,
            address      TEXT,
            phone        TEXT,
            skills       TEXT,
            hourly_rate  REAL,
            bio          TEXT,
            lat          REAL,
            lon          REAL,
            verified     TEXT DEFAULT "no",
            rating_sum   REAL DEFAULT 0,
            rating_count INTEGER DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES app_users(id)
        );
        ''')

        # 2. Schema migrations for existing databases
        # app_users: token_created_at
        au_cols = {r[1] for r in c.execute('PRAGMA table_info(app_users)')}
        if 'token_created_at' not in au_cols:
            c.execute('ALTER TABLE app_users ADD COLUMN token_created_at TEXT')

        # user_profiles: new columns
        up_cols = {r[1] for r in c.execute('PRAGMA table_info(user_profiles)')}
        for col, defn in [
            ('lat',          'REAL'),
            ('lon',          'REAL'),
            ('verified',     'TEXT DEFAULT "no"'),
            ('rating_sum',   'REAL DEFAULT 0'),
            ('rating_count', 'INTEGER DEFAULT 0'),
            ('birth_date',   'TEXT'),
            ('subdistrict',  'TEXT'),
            ('address',      'TEXT'),
        ]:
            if col not in up_cols:
                try:
                    c.execute(f'ALTER TABLE user_profiles ADD COLUMN {col} {defn}')
                except Exception:
                    pass

        # bookings: notes column
        bk_cols = {r[1] for r in c.execute('PRAGMA table_info(bookings)')}
        if 'notes' not in bk_cols:
            try:
                c.execute('ALTER TABLE bookings ADD COLUMN notes TEXT')
            except Exception:
                pass

        # 3. Upgrade legacy demo passwords → PBKDF2
        sha256_old = hashlib.sha256(b'demo1234').hexdigest()
        for old in (sha256_old, 'demo1234'):
            c.execute('UPDATE app_users SET password=? WHERE password=?',
                      (hash_pw('demo1234'), old))

        # 4. Seed demo accounts (only if table is empty)
        c.execute('SELECT COUNT(*) AS n FROM app_users')
        if c.fetchone()['n'] == 0:
            c.executemany(
                'INSERT INTO app_users (username, password, role, elderly_id) VALUES (?,?,?,?)',
                [
                    ('elder1', hash_pw('demo1234'), 'elderly', 0),
                    ('elder2', hash_pw('demo1234'), 'elderly', 1),
                    ('elder3', hash_pw('demo1234'), 'elderly', 2),
                ],
            )
            c.executemany(
                'INSERT INTO app_users (username, password, role, caretaker_id) VALUES (?,?,?,?)',
                [
                    ('care1', hash_pw('demo1234'), 'caretaker', 0),
                    ('care2', hash_pw('demo1234'), 'caretaker', 1),
                ],
            )
        conn.commit()


# ---------- Load seed data ----------
def load_seed_users():
    with open(DATA / 'users.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_seed_activities():
    with open(DATA / 'activities.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_seed_caretakers():
    with open(DATA / 'caretakers.csv', encoding='utf-8') as f:
        return list(csv.DictReader(f))


SEED_USERS: list       = []
SEED_ITEMS: list       = []
SEED_CARETAKERS: list  = []
USER_EMB               = None
ITEM_EMB               = None


def load_state():
    global SEED_USERS, SEED_ITEMS, SEED_CARETAKERS, USER_EMB, ITEM_EMB
    SEED_USERS       = load_seed_users()
    SEED_ITEMS       = load_seed_activities()
    SEED_CARETAKERS  = load_seed_caretakers()
    emb_path         = DATA / 'embeddings.npz'
    if emb_path.exists():
        npz      = np.load(emb_path)
        USER_EMB = npz['user_emb']
        ITEM_EMB = npz['item_emb']
    else:
        USER_EMB = None
        ITEM_EMB = None


# ---------- App ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_state()
    log_event(None, 'server_start', {'time': datetime.utcnow().isoformat()})
    yield


app = FastAPI(
    title='Promjai-Klaibaan API',
    description='พร้อมใจ ใกล้บ้าน — Mobile Recommender + Caretaker Matching',
    version='0.2.0',
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,   # Using Bearer tokens, not cookies
    allow_methods=['*'],
    allow_headers=['*'],
)


# ---------- Auth helpers ----------
def get_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith('Bearer '):
        raise HTTPException(401, 'Missing bearer token')
    return authorization[7:]


def get_user(token: str = Depends(get_token)):
    with db_conn() as conn:
        row = conn.execute('SELECT * FROM app_users WHERE token=?', (token,)).fetchone()
    if not row:
        raise HTTPException(401, 'Invalid token')
    user = dict(row)
    created_at = user.get('token_created_at')
    if created_at:
        age = datetime.utcnow() - datetime.fromisoformat(created_at)
        if age > timedelta(hours=TOKEN_TTL_HOURS):
            raise HTTPException(401, 'Token expired — please log in again')
    return user


# ---------- Logging ----------
def log_event(user_id, event_type, payload):
    ts   = datetime.utcnow().isoformat()
    line = json.dumps(
        {'ts': ts, 'user_id': user_id, 'event_type': event_type, 'payload': payload},
        ensure_ascii=False,
    )
    try:
        with open(LOG_DIR / 'events.jsonl', 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError:
        pass
    try:
        with db_conn() as conn:
            conn.execute(
                'INSERT INTO events (user_id, event_type, payload) VALUES (?,?,?)',
                (user_id, event_type, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
    except Exception:
        pass


# ---------- Age helper ----------
def _calc_age(birth_date_str: str) -> int:
    """Return age in years from a YYYY-MM-DD (CE) string."""
    from datetime import date as _date
    try:
        bd    = _date.fromisoformat(birth_date_str)
        today = _date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except Exception:
        return 0


# ---------- Pydantic models ----------
class LoginIn(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class WhoqolIn(BaseModel):
    physical:      float = Field(..., ge=0, le=100)
    psychological: float = Field(..., ge=0, le=100)
    social:        float = Field(..., ge=0, le=100)
    environment:   float = Field(..., ge=0, le=100)


class ActivityFeedbackIn(BaseModel):
    activity_id: int
    rating:      float = Field(..., ge=1, le=5)
    qol_delta:   Optional[float] = None
    note:        Optional[str]   = None


class RegisterIn(BaseModel):
    username:    str            = Field(..., min_length=3, max_length=50)
    password:    str            = Field(..., min_length=6, max_length=128)
    role:        str            = Field(..., pattern='^(elderly|caretaker)$')
    name:        str            = Field(..., min_length=1, max_length=100)
    birth_date:  str            = Field(..., pattern=r'^\d{4}-\d{2}-\d{2}$')  # CE: YYYY-MM-DD
    gender:      str            = Field(..., pattern='^(M|F|other)$')
    province:    str            = Field(default='')
    district:    str            = Field(default='')
    subdistrict: str            = Field(default='')
    address:     str            = Field(default='')
    phone:       Optional[str]  = None
    skills:      Optional[str]  = None    # "|"-separated, for caretakers
    hourly_rate: Optional[float] = Field(default=None, ge=0)
    lat:         Optional[float] = Field(default=None, ge=-90,  le=90)
    lon:         Optional[float] = Field(default=None, ge=-180, le=180)


class BookingIn(BaseModel):
    caretaker_id: int
    service_type: str  = Field(..., min_length=1, max_length=128)
    scheduled_at: Optional[str] = None   # ISO datetime string
    notes:        Optional[str] = Field(default=None, max_length=500)


class BookingStatusIn(BaseModel):
    status: str = Field(..., pattern='^(confirmed|completed|cancelled)$')


_ALLOWED_PROFILE_FIELDS = {
    'name', 'birth_date', 'gender', 'province', 'district', 'subdistrict', 'address',
    'phone', 'skills', 'hourly_rate', 'bio', 'lat', 'lon',
}


class ProfileUpdateIn(BaseModel):
    name:        Optional[str]   = Field(default=None, max_length=100)
    birth_date:  Optional[str]   = Field(default=None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    gender:      Optional[str]   = Field(default=None, pattern='^(M|F|other)$')
    province:    Optional[str]   = Field(default=None, max_length=100)
    district:    Optional[str]   = Field(default=None, max_length=100)
    subdistrict: Optional[str]   = Field(default=None, max_length=100)
    address:     Optional[str]   = Field(default=None, max_length=500)
    phone:       Optional[str]   = Field(default=None, max_length=30)
    skills:      Optional[str]   = Field(default=None, max_length=500)
    hourly_rate: Optional[float] = Field(default=None, ge=0)
    bio:         Optional[str]   = Field(default=None, max_length=1000)
    lat:         Optional[float] = Field(default=None, ge=-90,  le=90)
    lon:         Optional[float] = Field(default=None, ge=-180, le=180)


# ---------- Endpoints ----------
@app.get('/health')
def health():
    with db_conn() as conn:
        n_reg_caretakers = conn.execute(
            "SELECT COUNT(*) AS n FROM app_users WHERE role='caretaker' AND caretaker_id IS NULL"
        ).fetchone()['n']
    return {
        'status':              'ok',
        'time':                datetime.utcnow().isoformat(),
        'embeddings_loaded':   USER_EMB is not None,
        'n_seed_users':        len(SEED_USERS),
        'n_seed_activities':   len(SEED_ITEMS),
        'n_seed_caretakers':   len(SEED_CARETAKERS),
        'n_reg_caretakers':    n_reg_caretakers,
    }


@app.get('/metrics')
def get_metrics():
    """Returns the model metrics (RMSE/NDCG/HR) for the demo card."""
    p = ML / 'metrics.json'
    if p.exists():
        return json.loads(p.read_text(encoding='utf-8'))
    return {'note': 'No metrics yet — run ml/train_numpy.py'}


# ── Auth ────────────────────────────────────────────────────────────────────

@app.post('/auth/login')
def login(body: LoginIn, request: Request):
    ip = request.client.host if request.client else 'unknown'
    # Key = username:IP so brute-forcing one account doesn't block others from same IP
    _check_rate_limit(f'{body.username}:{ip}')
    with db_conn() as conn:
        row = conn.execute(
            'SELECT * FROM app_users WHERE username=?', (body.username,)
        ).fetchone()
        if not row or not _verify_pw(body.password, row['password']):
            raise HTTPException(401, 'Invalid credentials')
        # Opportunistic upgrade: re-hash legacy SHA-256 on successful login
        if not row['password'].startswith(_PW_PREFIX):
            conn.execute('UPDATE app_users SET password=? WHERE id=?',
                         (hash_pw(body.password), row['id']))
        token = secrets.token_urlsafe(24)
        now   = datetime.utcnow().isoformat()
        conn.execute(
            'UPDATE app_users SET token=?, token_created_at=? WHERE id=?',
            (token, now, row['id']),
        )
        conn.commit()
        user = dict(conn.execute('SELECT * FROM app_users WHERE id=?', (row['id'],)).fetchone())
    expires_at = (datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)).isoformat()
    log_event(user['id'], 'login', {'role': user['role']})
    return {
        'token':        token,
        'expires_at':   expires_at,
        'role':         user['role'],
        'username':     user['username'],
        'elderly_id':   user['elderly_id'],
        'caretaker_id': user['caretaker_id'],
    }


@app.post('/auth/register', status_code=201)
def register(body: RegisterIn):
    # Auto-fill lat/lon from province if not explicitly provided
    lat, lon = body.lat, body.lon
    if (lat is None or lon is None) and body.province:
        coords = PROVINCE_COORDS.get(body.province.strip())
        if coords:
            lat, lon = coords

    with db_conn() as conn:
        if conn.execute('SELECT id FROM app_users WHERE username=?',
                        (body.username,)).fetchone():
            raise HTTPException(409, 'Username already taken')
        cur = conn.execute(
            'INSERT INTO app_users (username, password, role) VALUES (?,?,?)',
            (body.username, hash_pw(body.password), body.role),
        )
        uid = cur.lastrowid
        conn.execute(
            '''INSERT INTO user_profiles
               (user_id, name, birth_date, gender, province, district, subdistrict, address,
                phone, skills, hourly_rate, lat, lon)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (uid, body.name, body.birth_date, body.gender,
             body.province, body.district, body.subdistrict, body.address,
             body.phone, body.skills, body.hourly_rate, lat, lon),
        )
        conn.commit()
    log_event(uid, 'register', {'role': body.role, 'province': body.province})
    return {'ok': True, 'user_id': uid, 'role': body.role}


@app.post('/auth/logout')
def logout(user=Depends(get_user)):
    with db_conn() as conn:
        conn.execute(
            'UPDATE app_users SET token=NULL, token_created_at=NULL WHERE id=?',
            (user['id'],),
        )
        conn.commit()
    log_event(user['id'], 'logout', {})
    return {'ok': True}


@app.get('/me')
def me(user=Depends(get_user)):
    profile = {'username': user['username'], 'role': user['role']}
    if user['role'] == 'elderly':
        idx = user.get('elderly_id')
        if idx is not None and 0 <= idx < len(SEED_USERS):
            seed = SEED_USERS[idx]
            profile.update({
                'name':     seed['name'],
                'age':      int(seed['age']),
                'gender':   seed['gender'],
                'province': seed['province'],
                'district': seed['district'],
                'tech_literacy': float(seed['tech_literacy']),
                'baseline_qol': {
                    'physical':      float(seed['baseline_qol_physical']),
                    'psychological': float(seed['baseline_qol_psychological']),
                    'social':        float(seed['baseline_qol_social']),
                    'environment':   float(seed['baseline_qol_environment']),
                },
            })
        else:
            with db_conn() as conn:
                p = conn.execute('SELECT * FROM user_profiles WHERE user_id=?',
                                 (user['id'],)).fetchone()
            if p:
                pd = dict(p)
                if pd.get('birth_date'):
                    pd['age'] = _calc_age(pd['birth_date'])
                profile.update(pd)
    elif user['role'] == 'caretaker':
        idx = user.get('caretaker_id')
        if idx is not None and 0 <= idx < len(SEED_CARETAKERS):
            c = SEED_CARETAKERS[idx]
            profile.update({
                'name':           c['name'],
                'age':            int(c['age']),
                'gender':         c['gender'],
                'rating':         float(c['rating']),
                'verified':       c['verified'],
                'skills':         c['skills'].split('|'),
                'hourly_rate':    float(c['hourly_rate']),
                'province':       c['province'],
                'district':       c['district'],
                'completed_jobs': int(c['completed_jobs']),
            })
        else:
            with db_conn() as conn:
                p = conn.execute('SELECT * FROM user_profiles WHERE user_id=?',
                                 (user['id'],)).fetchone()
            if p:
                pd = dict(p)
                count  = pd.get('rating_count') or 0
                rating = round(pd['rating_sum'] / count, 2) if count > 0 else None
                pd['rating'] = rating
                pd['skills'] = pd.get('skills', '').split('|') if pd.get('skills') else []
                if pd.get('birth_date'):
                    pd['age'] = _calc_age(pd['birth_date'])
                profile.update(pd)
    return profile


# ── Profile update ──────────────────────────────────────────────────────────

@app.put('/profile')
def update_profile(body: ProfileUpdateIn, user=Depends(get_user)):
    updates = {
        k: v
        for k, v in body.model_dump(exclude_none=True).items()
        if k in _ALLOWED_PROFILE_FIELDS
    }
    if not updates:
        raise HTTPException(400, 'ไม่มีข้อมูลที่ต้องการอัปเดต')

    # Auto-fill coordinates from province if province changed but lat/lon not provided
    if 'province' in updates and 'lat' not in updates:
        coords = PROVINCE_COORDS.get(updates['province'].strip())
        if coords:
            updates['lat'], updates['lon'] = coords

    cols = ', '.join(f'{k}=?' for k in updates)
    vals = list(updates.values()) + [user['id']]
    with db_conn() as conn:
        conn.execute('INSERT OR IGNORE INTO user_profiles (user_id) VALUES (?)', (user['id'],))
        conn.execute(f'UPDATE user_profiles SET {cols} WHERE user_id=?', vals)
        conn.commit()
    log_event(user['id'], 'profile_update', {'fields': list(updates.keys())})
    return {'ok': True, 'updated': list(updates.keys())}


# ── Recommendations ─────────────────────────────────────────────────────────

def compute_recs(user_idx: int, top_k: int = 8):
    if USER_EMB is None or ITEM_EMB is None:
        idx = list(range(len(SEED_ITEMS)))
        np.random.shuffle(idx)
        return [(i, 0.5, []) for i in idx[:top_k]]
    u_vec  = USER_EMB[user_idx]
    norm_u = u_vec / (np.linalg.norm(u_vec) + 1e-9)
    norm_i = ITEM_EMB / (np.linalg.norm(ITEM_EMB, axis=1, keepdims=True) + 1e-9)
    sims   = norm_i @ norm_u
    benefit = np.array([float(it['cognitive_benefit']) for it in SEED_ITEMS])
    score   = 0.7 * sims + 0.3 * benefit
    order   = np.argsort(-score)[:top_k]
    return [(int(i), float(score[i]), []) for i in order]


@app.get('/recommendations')
def recommendations(top_k: int = 8, user=Depends(get_user)):
    if user['role'] != 'elderly':
        raise HTTPException(403, 'Recommendations are for elderly users')
    if user['elderly_id'] is None:
        raise HTTPException(400, 'No seed elderly_id linked to account')
    top_k = min(max(1, top_k), 50)
    recs  = compute_recs(user['elderly_id'], top_k)
    out   = []
    for aid, score, _ in recs:
        a = SEED_ITEMS[aid]
        out.append({
            'activity_id':      aid,
            'name':             a['name_th'],
            'category':         a['category'],
            'cognitive_benefit': float(a['cognitive_benefit']),
            'difficulty':       a['difficulty'],
            'duration_min':     int(a['duration_min']),
            'description':      a['description'],
            'score':            round(score, 4),
        })
    log_event(user['id'], 'recommendations_request', {'top_k': top_k, 'count': len(out)})
    return {'recommendations': out}


_CATEGORY_TH = {
    'exercise':  'ออกกำลังกาย',
    'cognitive': 'ฝึกสมอง',
    'social':    'กิจกรรมกลุ่ม',
    'art':       'ศิลปะ',
    'music':     'ดนตรี',
    'nature':    'ธรรมชาติ',
    'cooking':   'ทำอาหาร',
    'games':     'เกม',
    'reading':   'อ่านหนังสือ',
    'spiritual': 'จิตใจ',
}
_DIFFICULTY_TH = {'easy': 'ง่าย', 'medium': 'ปานกลาง', 'hard': 'ท้าทาย'}

def _benefit_label(val: float) -> str:
    if val >= 0.45: return 'สูงมาก — ช่วยกระตุ้นสมองได้ดีเยี่ยม'
    if val >= 0.30: return 'ดี — ช่วยฝึกความจำและสมาธิ'
    return 'ปานกลาง — เหมาะสำหรับการผ่อนคลาย'

def _sim_label(sim: float) -> str:
    if sim >= 0.65: return 'เหมาะกับคุณมาก — ตรงกับกิจกรรมที่คุณชอบ'
    if sim >= 0.40: return 'น่าสนใจ — ใกล้เคียงกับสิ่งที่คุณเคยทำ'
    if sim >= 0.15: return 'น่าลอง — อาจค้นพบสิ่งใหม่ที่คุณชอบ'
    return 'แนะนำให้ลองดู — อาจเป็นประสบการณ์ใหม่ที่ดี'

@app.get('/explain/{activity_id}')
def explain(activity_id: int, user=Depends(get_user)):
    if user['role'] != 'elderly':
        raise HTTPException(403, 'Forbidden')
    if activity_id < 0 or activity_id >= len(SEED_ITEMS):
        raise HTTPException(404, 'Unknown activity')
    a      = SEED_ITEMS[activity_id]
    me_idx = user['elderly_id']
    if me_idx is None or me_idx < 0 or me_idx >= len(SEED_USERS):
        raise HTTPException(400, 'Invalid user profile')
    seed     = SEED_USERS[me_idx]
    cat_th   = _CATEGORY_TH.get(a['category'], a['category'])
    diff_th  = _DIFFICULTY_TH.get(a['difficulty'], a['difficulty'])
    ben_lbl  = _benefit_label(float(a['cognitive_benefit']))
    expl = [
        f"ระบบเลือกกิจกรรม '{a['name_th']}' มาให้คุณโดยเฉพาะ เพราะ:",
        f"📂 เป็นกิจกรรมประเภท{cat_th} เหมาะสำหรับผู้สูงอายุ",
        f"💪 ระดับ{diff_th} — เหมาะกับช่วงวัย {seed['age']} ปี",
        f"🧠 ประโยชน์ต่อสมอง: {ben_lbl}",
        f"⏱ ใช้เวลาเพียง {a['duration_min']} นาที ทำได้ที่บ้าน",
    ]
    if USER_EMB is not None:
        u_vec = USER_EMB[me_idx]
        i_vec = ITEM_EMB[activity_id]
        sim   = float(np.dot(u_vec, i_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(i_vec) + 1e-9))
        expl.append(f"❤️ ความเหมาะสมกับคุณ: {_sim_label(sim)}")
    return {'activity': a['name_th'], 'explanation': expl}


# ── Activity feedback ────────────────────────────────────────────────────────

@app.post('/feedback')
def feedback(body: ActivityFeedbackIn, user=Depends(get_user)):
    with db_conn() as conn:
        conn.execute(
            'INSERT INTO activity_logs (user_id, activity_id, rating, qol_delta) VALUES (?,?,?,?)',
            (user['id'], body.activity_id, body.rating, body.qol_delta),
        )
        conn.commit()
    log_event(user['id'], 'activity_feedback', body.model_dump())
    return {'ok': True}


# ── WHOQOL tracking ──────────────────────────────────────────────────────────

@app.post('/whoqol')
def whoqol_save(body: WhoqolIn, user=Depends(get_user)):
    with db_conn() as conn:
        conn.execute(
            'INSERT INTO qol_history (user_id, physical, psychological, social, environment) VALUES (?,?,?,?,?)',
            (user['id'], body.physical, body.psychological, body.social, body.environment),
        )
        conn.commit()
    log_event(user['id'], 'whoqol_record', body.model_dump())
    overall = (body.physical + body.psychological + body.social + body.environment) / 4
    return {'ok': True, 'overall': round(overall, 2)}


@app.get('/whoqol/history')
def whoqol_history(user=Depends(get_user)):
    with db_conn() as conn:
        rows = conn.execute(
            'SELECT * FROM qol_history WHERE user_id=? ORDER BY recorded_at',
            (user['id'],),
        ).fetchall()
    return [dict(r) for r in rows]


# ── Caretaker matching ───────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@app.get('/caretakers/match')
def caretakers_match(top_k: int = 5, max_km: float = 30.0, user=Depends(get_user)):
    if user['role'] != 'elderly':
        raise HTTPException(403, 'Only elderly users can match caretakers')

    # Determine requester location
    me_lat = me_lon = None
    me_idx = user.get('elderly_id')
    if me_idx is not None and 0 <= me_idx < len(SEED_USERS):
        seed   = SEED_USERS[me_idx]
        me_lat = float(seed['lat'])
        me_lon = float(seed['lon'])
    else:
        with db_conn() as conn:
            p = conn.execute(
                'SELECT lat, lon FROM user_profiles WHERE user_id=?', (user['id'],)
            ).fetchone()
        if p and p['lat'] is not None and p['lon'] is not None:
            me_lat = float(p['lat'])
            me_lon = float(p['lon'])

    if me_lat is None:
        raise HTTPException(
            400,
            'ยังไม่ได้ตั้งค่าที่อยู่ — กรุณาอัปเดตจังหวัดในโปรไฟล์ก่อน'
        )

    top_k      = min(max(1, top_k), 50)
    candidates = []

    # --- Seed caretakers ---
    for c in SEED_CARETAKERS:
        d = haversine_km(me_lat, me_lon, float(c['lat']), float(c['lon']))
        if d > max_km:
            continue
        verified_bonus = 0.4 if c['verified'] == 'yes' else 0
        score = (
            (float(c['rating']) / 5) * 0.5
            + verified_bonus
            + max(0, (max_km - d) / max_km) * 0.4
        )
        candidates.append({
            'caretaker_id': int(c['caretaker_id']),
            'source':       'seed',
            'name':         c['name'],
            'rating':       float(c['rating']),
            'verified':     c['verified'],
            'skills':       c['skills'].split('|'),
            'hourly_rate':  float(c['hourly_rate']),
            'distance_km':  round(d, 2),
            'province':     c['province'],
            'district':     c['district'],
            'lat':          float(c['lat']) if c.get('lat') else None,
            'lon':          float(c['lon']) if c.get('lon') else None,
            'match_score':  round(score, 3),
        })

    # --- Registered caretakers with known location ---
    with db_conn() as conn:
        reg_rows = conn.execute(
            '''SELECT au.id, up.name, up.skills, up.hourly_rate,
                      up.province, up.district, up.lat, up.lon,
                      up.verified, up.rating_sum, up.rating_count
               FROM app_users au
               JOIN user_profiles up ON au.id = up.user_id
               WHERE au.role = "caretaker"
                 AND up.lat  IS NOT NULL
                 AND up.lon  IS NOT NULL'''
        ).fetchall()

    for r in reg_rows:
        d_km = haversine_km(me_lat, me_lon, float(r['lat']), float(r['lon']))
        if d_km > max_km:
            continue
        count    = r['rating_count'] or 0
        rating   = (r['rating_sum'] / count) if count > 0 else 3.0
        verified = r['verified'] or 'no'
        vbonus   = 0.4 if verified == 'yes' else 0
        score    = (
            (min(rating, 5) / 5) * 0.5
            + vbonus
            + max(0, (max_km - d_km) / max_km) * 0.4
        )
        candidates.append({
            'caretaker_id': -(r['id']),      # negative → registered user
            'source':       'registered',
            'name':         r['name'] or '-',
            'rating':       round(rating, 2),
            'verified':     verified,
            'skills':       (r['skills'] or '').split('|') if r['skills'] else [],
            'hourly_rate':  r['hourly_rate'] or 0,
            'distance_km':  round(d_km, 2),
            'province':     r['province'] or '',
            'district':     r['district'] or '',
            'lat':          float(r['lat']),
            'lon':          float(r['lon']),
            'match_score':  round(score, 3),
        })

    candidates.sort(key=lambda x: -x['match_score'])
    result = candidates[:top_k]
    log_event(user['id'], 'caretaker_match', {'count': len(result), 'max_km': max_km})
    return {'matches': result, 'user_location': {'lat': me_lat, 'lon': me_lon}}


@app.get('/caretakers')
def list_all_caretakers(user=Depends(get_user)):
    """List all caretakers: seed (CSV) + registered (DB). Requires auth."""
    out = []

    # Seed caretakers
    for i, c in enumerate(SEED_CARETAKERS):
        out.append({
            'source':         'seed',
            'caretaker_id':   i,
            'name':           c['name'],
            'rating':         float(c['rating']),
            'verified':       c['verified'],
            'skills':         c['skills'].split('|'),
            'hourly_rate':    float(c['hourly_rate']),
            'province':       c['province'],
            'district':       c['district'],
            'completed_jobs': int(c['completed_jobs']),
        })

    # Registered caretakers
    with db_conn() as conn:
        rows = conn.execute(
            '''SELECT au.id, au.username, au.created_at,
                      up.name, up.age, up.gender,
                      up.province, up.district, up.phone, up.skills,
                      up.hourly_rate, up.verified, up.rating_sum, up.rating_count,
                      up.lat, up.lon
               FROM app_users au
               JOIN user_profiles up ON au.id = up.user_id
               WHERE au.role = "caretaker"
               ORDER BY au.created_at DESC'''
        ).fetchall()

    for r in rows:
        count  = r['rating_count'] or 0
        rating = round(r['rating_sum'] / count, 2) if count > 0 else None
        out.append({
            'source':       'registered',
            'caretaker_id': -(r['id']),
            'username':     r['username'],
            'name':         r['name'] or '-',
            'rating':       rating,
            'verified':     r['verified'] or 'no',
            'skills':       (r['skills'] or '').split('|') if r['skills'] else [],
            'hourly_rate':  r['hourly_rate'],
            'province':     r['province'] or '',
            'district':     r['district'] or '',
            'has_location': r['lat'] is not None,
            'registered_at': r['created_at'],
        })

    return {'caretakers': out, 'total': len(out)}


# ── Booking ──────────────────────────────────────────────────────────────────

def _eid_for_user(user: dict) -> int:
    """Return the elderly_id to store in bookings.elderly_id.
    Seed users use their positive CSV index; registered users use -(user_id).
    """
    idx = user.get('elderly_id')
    return idx if idx is not None else -(user['id'])


def _cid_for_user(user: dict) -> int:
    """Return the caretaker_id that identifies this caretaker in bookings."""
    idx = user.get('caretaker_id')
    return idx if idx is not None else -(user['id'])


@app.post('/bookings')
def create_booking(body: BookingIn, user=Depends(get_user)):
    if user['role'] != 'elderly':
        raise HTTPException(403, 'Only elderly users can book')
    eid = _eid_for_user(user)
    with db_conn() as conn:
        cur = conn.execute(
            "INSERT INTO bookings (elderly_id, caretaker_id, service_type, scheduled_at, notes, status)"
            " VALUES (?,?,?,?,?,'pending')",
            (eid, body.caretaker_id, body.service_type, body.scheduled_at, body.notes),
        )
        bid = cur.lastrowid
        conn.commit()
    log_event(user['id'], 'booking_create', body.model_dump())
    return {'booking_id': bid, 'status': 'pending'}


@app.get('/bookings')
def list_bookings(user=Depends(get_user)):
    with db_conn() as conn:
        if user['role'] == 'elderly':
            eid  = _eid_for_user(user)
            rows = conn.execute(
                'SELECT * FROM bookings WHERE elderly_id=? ORDER BY created_at DESC',
                (eid,),
            ).fetchall()
        else:
            my_cid = _cid_for_user(user)
            rows   = conn.execute(
                'SELECT * FROM bookings WHERE caretaker_id=? ORDER BY created_at DESC',
                (my_cid,),
            ).fetchall()

        out = []
        for r in rows:
            d   = dict(r)
            cid = d.get('caretaker_id')
            if cid is not None:
                if cid >= 0 and cid < len(SEED_CARETAKERS):
                    d['caretaker_name'] = SEED_CARETAKERS[cid]['name']
                elif cid < 0:
                    p = conn.execute(
                        'SELECT name FROM user_profiles WHERE user_id=?', (-cid,)
                    ).fetchone()
                    d['caretaker_name'] = p['name'] if p else f'ผู้ดูแล #{-cid}'
            eid_val = d.get('elderly_id')
            if eid_val is not None:
                if eid_val >= 0 and eid_val < len(SEED_USERS):
                    d['elderly_name'] = SEED_USERS[eid_val]['name']
                elif eid_val < 0:
                    p = conn.execute(
                        'SELECT name FROM user_profiles WHERE user_id=?', (-eid_val,)
                    ).fetchone()
                    d['elderly_name'] = p['name'] if p else f'ผู้ใช้ #{-eid_val}'
            out.append(d)
    return out


@app.patch('/bookings/{booking_id}/status')
def update_booking_status(booking_id: int, body: BookingStatusIn, user=Depends(get_user)):
    """Caretaker: confirmed / completed.  Either party: cancelled."""
    with db_conn() as conn:
        row = conn.execute('SELECT * FROM bookings WHERE id=?', (booking_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'Booking not found')
        row = dict(row)

        if user['role'] == 'caretaker':
            bid_stored = row['caretaker_id']
            if bid_stored is None:
                raise HTTPException(403, 'Not your booking')
            if bid_stored >= 0:
                # Seed caretaker: match by caretaker_id column
                if bid_stored != user.get('caretaker_id'):
                    raise HTTPException(403, 'Not your booking')
            else:
                # Registered caretaker: match by -(user_id)
                if bid_stored != -(user['id']):
                    raise HTTPException(403, 'Not your booking')

        elif user['role'] == 'elderly':
            my_eid = _eid_for_user(user)
            if row['elderly_id'] != my_eid:
                raise HTTPException(403, 'Not your booking')
            if body.status != 'cancelled':
                raise HTTPException(403, 'Elderly can only cancel bookings')

        conn.execute('UPDATE bookings SET status=? WHERE id=?', (body.status, booking_id))
        conn.commit()

    log_event(user['id'], 'booking_status_update',
              {'booking_id': booking_id, 'status': body.status})
    return {'booking_id': booking_id, 'status': body.status}


# ── Aggregate stats ──────────────────────────────────────────────────────────

@app.get('/stats/summary')
def stats_summary():
    with db_conn() as conn:
        n_users    = conn.execute('SELECT COUNT(*) AS n FROM app_users').fetchone()['n']
        n_logs     = conn.execute('SELECT COUNT(*) AS n FROM activity_logs').fetchone()['n']
        n_qol      = conn.execute('SELECT COUNT(*) AS n FROM qol_history').fetchone()['n']
        n_bookings = conn.execute('SELECT COUNT(*) AS n FROM bookings').fetchone()['n']
        n_events   = conn.execute('SELECT COUNT(*) AS n FROM events').fetchone()['n']
        avg_rating = conn.execute('SELECT AVG(rating) AS r FROM activity_logs').fetchone()['r']
    return {
        'app_users':           n_users,
        'activity_logs':       n_logs,
        'qol_records':         n_qol,
        'bookings':            n_bookings,
        'events':              n_events,
        'avg_activity_rating': round(avg_rating, 2) if avg_rating else None,
    }


# ── Static frontend ──────────────────────────────────────────────────────────

FRONTEND_DIR = ROOT / 'frontend'
if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

    @app.get('/')
    def index():
        f = FRONTEND_DIR / 'index.html'
        if f.exists():
            return FileResponse(str(f))
        return {'msg': 'frontend not found'}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
