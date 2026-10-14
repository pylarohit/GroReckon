import json

with open("app/src/main/assets/demo/demo_trip.json") as f:
    d = json.load(f)

for s in d["samples"][340:400:10]:
    print(f"t={s['timestampMs']/1000:.1f}s | acc=[{s['accelX']:.2f}, {s['accelY']:.2f}, {s['accelZ']:.2f}] | AI V={s['aiSpeedMps']:.2f} | Est V={s['speedMps']:.2f} | Est X={s['estX']:.1f}")
