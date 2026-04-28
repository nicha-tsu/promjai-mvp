# Pilot Protocol — กลุ่มทดลอง 5–10 ราย (เบื้องต้น)

## 1. กลุ่มเป้าหมาย
- ผู้สูงอายุ 60–80 ปี ที่ใช้สมาร์ทโฟนได้ระดับเบื้องต้น (LINE)
- ผู้ดูแลในชุมชน 2–3 คน (อสม. หรือ อาสาสมัคร)

## 2. Inclusion / Exclusion
**Inclusion:**
- ยินยอมเข้าร่วม (ลงนาม Consent Form)
- ใช้แอปได้อย่างน้อย 30 นาที/สัปดาห์
- มีสมาร์ทโฟน Android หรือ iOS

**Exclusion:**
- ภาวะสมองเสื่อมระยะปานกลาง-รุนแรง (MMSE < 18)
- พิการการมองเห็นรุนแรง
- ปฏิเสธการลงนาม Consent

## 3. Data Collection
| Time | Instrument | Purpose |
|---|---|---|
| Day 0 (Baseline) | WHOQOL-BREF (ในแอป), Demographics, MMSE | Baseline QoL |
| Day 7 | App usage logs, Activity ratings | Engagement |
| Day 30 | WHOQOL-BREF (Post 1), SUS, NPS | Mid-eval |
| Day 60 | WHOQOL-BREF (Post 2), Interview | Final |

## 4. Outcomes
**Primary:** ΔWHOQOL-BREF (Cohen's d, target ≥ 0.5)
**Secondary:**
- Active User Rate (target ≥ 60% Day-7, ≥ 40% Day-30)
- SUS Score (target ≥ 70)
- NPS (target ≥ +30)
- Recommendation acceptance rate (target ≥ 30%)

## 5. Statistical Plan
- Paired t-test (or Wilcoxon if non-normal) for ΔWHOQOL
- Descriptive stats for SUS, NPS
- Thematic analysis for interview

## 6. Ethics
- Submit IRB Exempt application (ม.ทักษิณ)
- Anonymize logs before analysis (replace user_id with hash)
- Right to withdraw at any time
- Data retention: 5 years per institutional policy

## 7. Analysis Pipeline
```bash
# After pilot: extract logs
sqlite3 promjai.db ".dump events" > events.sql
cp logs/events.jsonl events_raw.jsonl

# Analyze in Jupyter/Pandas
python scripts/analyze_pilot.py
```

(See `scripts/analyze_pilot.py` if needed — to be implemented per pilot.)
