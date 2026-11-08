import subprocess
import time

def thermals():
    fields = "temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    values = [s.strip() for s in result.stdout.splitlines()[0].split(",")]
    if len(values) != 3 or any(value not in ("Active", "Not Active") for value in values[1:]):
        raise RuntimeError("Thermal throttling status is unavailable; do not assume it is safe")
    return {"temperature_C": int(values[0]), "hw_thermal": values[1], "sw_thermal": values[2],
            "active": any(s == "Active" for s in values[1:]), "time": time.time()}
