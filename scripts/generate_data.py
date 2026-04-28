"""
Generate synthetic dataset matching the paper:
- 200 elderly users (age 60-85)
- 100 cognitive activities across 10 categories
- 5,000 timestamped interactions
- 600 social edges via age-similarity preferential attachment

Also generates 50 caretakers with location for the matching feature
(BMC: Caretaker Matching Service).

Outputs:
  data/users.csv
  data/activities.csv
  data/interactions.csv
  data/social_edges.csv
  data/caretakers.csv
"""
import os
import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Activities ----------
CATEGORIES = [
    ('exercise', 'การออกกำลังกายเบาๆ', 0.45),
    ('cognitive', 'ฝึกความจำ/สมอง', 0.50),
    ('social', 'กิจกรรมกลุ่ม', 0.35),
    ('art', 'ศิลปะและงานฝีมือ', 0.30),
    ('music', 'ดนตรีและการขับร้อง', 0.32),
    ('mindful', 'สมาธิและผ่อนคลาย', 0.40),
    ('nutrition', 'โภชนาการ/ทำอาหาร', 0.38),
    ('outdoor', 'กิจกรรมกลางแจ้ง', 0.42),
    ('learning', 'เรียนรู้ทักษะใหม่', 0.48),
    ('volunteer', 'จิตอาสาในชุมชน', 0.36),
]

ACTIVITY_NAMES = {
    'exercise': ['ไทเก็ก', 'โยคะเบาๆ', 'รำพื้นบ้าน', 'เดินเร็วในสวน', 'ออกกำลังบนเก้าอี้',
                 'ยืดเหยียดยามเช้า', 'ฟิตเนสเบาๆ', 'รำมวยจีน', 'แอโรบิกผู้สูงวัย', 'เดินวันละ 1 กม.'],
    'cognitive': ['เกมจับคู่ภาพ', 'ครอสเวิร์ดไทย', 'ซูโดกุ', 'หมากรุก', 'หมากกระดาน',
                  'จดจำลำดับเลข', 'เกมความจำ', 'อ่านนิทาน-สรุป', 'ทายปัญหา', 'ฝึกคำนวณ'],
    'social': ['สนทนาวงแชร์', 'ชมรมผู้สูงอายุ', 'ตลาดนัดสุขภาพ', 'งานบุญในวัด', 'ปาร์ตี้น้ำชา',
               'พบปะเพื่อนเก่า', 'งานบรรยายสุขภาพ', 'ทริปหนึ่งวัน', 'ปลูกผักร่วม', 'ทำขนมร่วมกัน'],
    'art': ['วาดภาพระบายสี', 'ปั้นดินไทย', 'ถักโครเชต์', 'จักสาน', 'ทำเครื่องประดับ',
            'ทำการ์ดอวยพร', 'แต่งสวนถาด', 'ทำดอกไม้กระดาษ', 'งานปะติด', 'เพ้นท์เซรามิค'],
    'music': ['ร้องเพลงเก่า', 'เล่นซอ', 'เล่นขลุ่ย', 'คาราโอเกะ', 'เคาะจังหวะ',
              'ฟังเพลงร่วม', 'ลีลาศ', 'ร้องเพลงลูกทุ่ง', 'เล่นกลอง', 'ทำกล่องดนตรี'],
    'mindful': ['สมาธิ 10 นาที', 'หายใจ 4-7-8', 'นั่งสมาธิเดิน', 'โยคะนิทรา', 'จิตศึกษา',
                'สวดมนต์', 'ฟังธรรม', 'ผ่อนคลายกล้ามเนื้อ', 'จดบันทึกขอบคุณ', 'ดื่มชาอย่างมีสติ'],
    'nutrition': ['ทำส้มตำไม่เผ็ด', 'ทำต้มจืด', 'ปรับสูตรลดเค็ม', 'ทำสลัดผัก', 'ทำขนมไทย',
                  'อ่านฉลากโภชนาการ', 'ทำน้ำสมุนไพร', 'ทำกับข้าวลดน้ำตาล', 'ปลูกผักสวนครัว', 'ทำข้าวต้มกุ้ง'],
    'outdoor': ['เดินตลาดเช้า', 'เก็บผักสวน', 'รดน้ำต้นไม้', 'ไปวัดทำบุญ', 'นั่งฟังนกที่สวน',
                'ปั่นจักรยานเบาๆ', 'เดินรอบหมู่บ้าน', 'ตกปลา', 'พายเรือเล่น', 'ปิคนิคในชุมชน'],
    'learning': ['ใช้สมาร์ทโฟนเบื้องต้น', 'LINE ส่งข้อความ', 'อ่านข่าวสุขภาพ', 'เรียนภาษาอังกฤษพื้นฐาน',
                 'หัดถ่ายรูป', 'ใช้ YouTube', 'การออม', 'ภาษีสำหรับผู้สูงวัย', 'เรียนรู้สมุนไพร', 'หัดทำ vlog'],
    'volunteer': ['ช่วยงานวัด', 'อ่านหนังสือให้เด็ก', 'สอนงานฝีมือ', 'ดูแลสวนสาธารณะ', 'อบรมเล่นบอร์ดเกม',
                  'ช่วยงาน อสม.', 'แจกอาหารผู้ป่วย', 'สอนทำขนม', 'ช่วยงานโรงเรียน', 'ร่วมเก็บขยะชุมชน'],
}

with open(os.path.join(OUT_DIR, 'activities.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['activity_id', 'name_th', 'category', 'cognitive_benefit',
                'difficulty', 'duration_min', 'description'])
    aid = 0
    for cat, cat_th, base_benefit in CATEGORIES:
        names = ACTIVITY_NAMES[cat]
        for n in names:
            difficulty = random.choice(['easy', 'easy', 'medium', 'medium', 'hard'])
            duration = random.choice([15, 20, 30, 30, 45, 60])
            benefit = round(base_benefit + random.uniform(-0.08, 0.08), 3)
            desc = f'{n} เป็นกิจกรรมประเภท{cat_th} ใช้เวลาประมาณ {duration} นาที'
            w.writerow([aid, n, cat, benefit, difficulty, duration, desc])
            aid += 1
print(f'  -> activities.csv ({aid} rows)')

# ---------- Users ----------
PROVINCES = ['พัทลุง', 'สงขลา', 'ตรัง', 'นครศรีธรรมราช', 'สุราษฎร์ธานี']
DISTRICTS_BY_PROV = {
    'พัทลุง': ['ป่าพะยอม', 'ควนขนุน', 'เมืองพัทลุง', 'ศรีบรรพต'],
    'สงขลา': ['หาดใหญ่', 'เมืองสงขลา', 'สะเดา', 'ระโนด'],
    'ตรัง': ['เมืองตรัง', 'ห้วยยอด', 'กันตัง', 'ย่านตาขาว'],
    'นครศรีธรรมราช': ['เมืองนคร', 'ทุ่งสง', 'ปากพนัง', 'สิชล'],
    'สุราษฎร์ธานี': ['เมืองสุราษฎร์', 'เกาะสมุย', 'ไชยา', 'พุนพิน'],
}

with open(os.path.join(OUT_DIR, 'users.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'name', 'age', 'gender', 'tech_literacy',
                'baseline_qol_physical', 'baseline_qol_psychological',
                'baseline_qol_social', 'baseline_qol_environment',
                'province', 'district', 'lat', 'lon'])
    for uid in range(200):
        age = int(random.gauss(70, 6))
        age = max(60, min(85, age))
        gender = random.choice(['M', 'F', 'F'])  # slight female skew
        tech = round(random.uniform(0.1, 1.0), 2)
        # baseline QoL ~ N(55, 10) by domain
        q_phys = round(random.gauss(55, 10), 1)
        q_psy = round(random.gauss(55, 10), 1)
        q_soc = round(random.gauss(53, 11), 1)
        q_env = round(random.gauss(57, 9), 1)
        prov = random.choices(PROVINCES, weights=[0.40, 0.25, 0.15, 0.12, 0.08])[0]
        dist = random.choice(DISTRICTS_BY_PROV[prov])
        # latlon clustered around southern Thailand
        lat = round(7.5 + random.uniform(-1.0, 1.5), 5)
        lon = round(100.0 + random.uniform(-1.5, 1.0), 5)
        name = f'ผู้ใช้ {uid+1:03d}'
        w.writerow([uid, name, age, gender, tech,
                    q_phys, q_psy, q_soc, q_env,
                    prov, dist, lat, lon])
print(f'  -> users.csv (200 rows)')

# ---------- Social edges (age-similarity preferential attachment) ----------
import sys
# load user ages
ages = {}
provs = {}
with open(os.path.join(OUT_DIR, 'users.csv'), encoding='utf-8') as f:
    r = csv.DictReader(f)
    for row in r:
        ages[int(row['user_id'])] = int(row['age'])
        provs[int(row['user_id'])] = row['province']

edges = set()
target = 600
attempts = 0
while len(edges) < target and attempts < 50000:
    u = random.randrange(200)
    v = random.randrange(200)
    if u == v:
        attempts += 1
        continue
    age_diff = abs(ages[u] - ages[v])
    same_prov = (provs[u] == provs[v])
    # P(connect) higher if age_diff small + same province
    p = math.exp(-age_diff / 5.0) * (1.5 if same_prov else 1.0) * 0.3
    if random.random() < p:
        edges.add(tuple(sorted((u, v))))
    attempts += 1

with open(os.path.join(OUT_DIR, 'social_edges.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_a', 'user_b'])
    for a, b in edges:
        w.writerow([a, b])
print(f'  -> social_edges.csv ({len(edges)} edges)')

# ---------- Interactions ----------
# For each user, sample 20-30 activities (preferential to category they like)
with open(os.path.join(OUT_DIR, 'activities.csv'), encoding='utf-8') as f:
    activities = list(csv.DictReader(f))

# assign each user a "preferred category" with weight
user_pref = {}
for uid in range(200):
    user_pref[uid] = {cat: random.uniform(0.3, 1.0) for cat, _, _ in CATEGORIES}
    # boost 2-3 categories
    fav = random.sample([c for c, _, _ in CATEGORIES], 3)
    for c in fav:
        user_pref[uid][c] += 1.0

start_date = datetime(2025, 1, 1)
with open(os.path.join(OUT_DIR, 'interactions.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['user_id', 'activity_id', 'rating', 'qol_delta', 'timestamp'])
    cnt = 0
    for uid in range(200):
        n = random.randint(20, 30)
        # weighted sampling
        weights = [user_pref[uid][a['category']] * float(a['cognitive_benefit'])
                   for a in activities]
        total = sum(weights)
        sampled_ids = set()
        while len(sampled_ids) < n:
            r = random.random() * total
            cum = 0
            for i, w_ in enumerate(weights):
                cum += w_
                if cum >= r:
                    sampled_ids.add(i)
                    break
        for aidx in sampled_ids:
            a = activities[aidx]
            base = float(a['cognitive_benefit'])
            pref = user_pref[uid][a['category']]
            rating = base * 4 + (pref - 1) * 1.5 + random.uniform(-0.5, 0.5)
            rating = max(1.0, min(5.0, rating))
            # qol delta — gain a bit if rating high
            qol_delta = (rating - 3.0) * random.uniform(0.5, 1.5) + random.uniform(-0.3, 0.3)
            ts = start_date + timedelta(days=random.randint(0, 90),
                                        minutes=random.randint(0, 1440))
            w.writerow([uid, aidx, round(rating, 2), round(qol_delta, 2),
                        ts.isoformat()])
            cnt += 1
            if cnt >= 5000:
                break
        if cnt >= 5000:
            break
print(f'  -> interactions.csv ({cnt} rows)')

# ---------- Caretakers ----------
CARE_FIRSTNAMES = ['สมศรี', 'มาลี', 'สุดา', 'วาสนา', 'น้ำฝน', 'กชกร',
                   'ทศพล', 'ภูริ', 'อนุชา', 'ธนกร', 'พิชญ์', 'นภัส']
SKILLS = ['ดูแลผู้สูงอายุทั่วไป', 'พยาบาลพื้นฐาน', 'นวดแผนไทย', 'ทำอาหารผู้สูงวัย',
         'เพื่อนสนทนา', 'พาออกกำลังกาย', 'ขับรถพาไปหาหมอ', 'ทำความสะอาด']

with open(os.path.join(OUT_DIR, 'caretakers.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['caretaker_id', 'name', 'age', 'gender', 'rating', 'verified',
                'skills', 'hourly_rate', 'province', 'district', 'lat', 'lon',
                'completed_jobs'])
    for cid in range(50):
        first = random.choice(CARE_FIRSTNAMES)
        name = f'{first} (ID-{cid:03d})'
        age = random.randint(20, 55)
        gender = random.choice(['F', 'F', 'F', 'M'])
        rating = round(random.uniform(3.5, 5.0), 1)
        verified = random.choice(['yes', 'yes', 'yes', 'pending'])
        n_skills = random.randint(2, 4)
        skills = '|'.join(random.sample(SKILLS, n_skills))
        rate = random.choice([150, 180, 200, 220, 250, 300])
        prov = random.choices(PROVINCES, weights=[0.40, 0.25, 0.15, 0.12, 0.08])[0]
        dist = random.choice(DISTRICTS_BY_PROV[prov])
        lat = round(7.5 + random.uniform(-1.0, 1.5), 5)
        lon = round(100.0 + random.uniform(-1.5, 1.0), 5)
        completed = random.randint(3, 80)
        w.writerow([cid, name, age, gender, rating, verified,
                    skills, rate, prov, dist, lat, lon, completed])
print(f'  -> caretakers.csv (50 rows)')

print('\n✅ Synthetic data generation complete')
