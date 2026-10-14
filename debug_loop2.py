import json
with open("app/src/main/assets/demo/demo_trip.json") as f:
    d = json.load(f)

for s in d["samples"][348:356]:
    print(f"t={s['timestampMs']/1000}s | aiSpeed={s['aiSpeedMps']}")
