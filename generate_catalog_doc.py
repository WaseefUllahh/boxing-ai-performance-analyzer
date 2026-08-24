import json

with open("fight2_candidates.json") as f:
    events = json.load(f)

lines = []
lines.append("# Candidate Review Catalog — External Generalization Fight (`data/fight2.mp4`)")
lines.append("")
lines.append("**Video Specifications**: 1920x1080 @ 25.00 FPS | Total Frames: 5,254 (210.12s)")
lines.append("**Fighters**: Canelo Alvarez vs. Gennady Golovkin II")
lines.append("**Scale Manager**: RobustScaleManager (30-frame frame-age expiration, anthropometric torso-ratio gating, 80th-percentile scale)")
lines.append("")
lines.append("| Event # | Timestamp | Frame | Attacker | Strike Type | Hand | Target | Pred. Outcome | Confidence | Dynamic Physical Diagnostics |")
lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

for i, e in enumerate(events):
    eid = i + 1
    t = f"{e['timestamp']:.2f}s"
    f = e['frame']
    att = f"Fighter {e['fighter_id']}"
    act = e['action']
    h = e['hand'].capitalize()
    tz = e['target_zone']
    pred = e['predicted_outcome']
    conf = f"{e['confidence']:.2f}"
    reason = e['trigger_reason']
    lines.append(f"| #{eid:03d} | {t} | {f} | {att} | {act} | {h} | {tz} | **{pred}** | {conf} | `{reason}` |")

out_path = r"C:\Users\PMLS\.gemini\antigravity-ide\brain\c4677914-727f-494f-a6de-c8c3fda73f46\candidate_review_catalog.md"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Successfully wrote {len(events)} events to candidate_review_catalog.md")
