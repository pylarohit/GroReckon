import json

with open("app/src/main/assets/demo/demo_trip.json", "r") as f:
    data = json.load(f)

samples = data["samples"]
for i in range(350, 750, 50):
    s = samples[i]
    print(f"t={s['timestampMs']/1000:.1f}s | True V={s['speedMps']:.1f} | AI V={s['aiSpeedMps']:.1f} | Err={s['errorM']:.1f}m | X={s['estX']:.1f} (True {data['samples'][i]['trueLat']})")
