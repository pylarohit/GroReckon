import json
with open("app/src/main/assets/demo/demo_trip.json") as f:
    d = json.load(f)

for s in d["samples"][340:370:5]:
    print(f"t={s['timestampMs']/1000}s | estX={s['estX']} | estY={s['estY']} | heading={s['headingDeg']} | mode={s['mode']}")
