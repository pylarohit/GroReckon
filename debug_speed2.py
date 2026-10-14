import json

with open("app/src/main/assets/demo/demo_trip.json") as f:
    d = json.load(f)

for s in d["samples"][350:370:2]:
    print(f"t={s['timestampMs']/1000:.1f}s | AI Speed={s['aiSpeedMps']:.2f} m/s | True Speed={s['speedKmh']/3.6:.2f} m/s")
