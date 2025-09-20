# Hue sound-reactive lights — simple, show-first version (1-second windows)
#
# What it does:
# - Listens in 1-second chunks.
# - Compares the current second's average loudness to the previous second.
#   • If it's louder → lights shift toward warm/red.
#   • If it's quieter → lights shift toward cool/blue.
# - Brightness follows the absolute loudness (quiet = dim, talking = brighter).
# - One Hue REST update per second with a gentle transition (looks smooth).
# - Beeps indicate up/down changes; prints dB and Δ each second; shows an ASCII ear on start.
# - On Ctrl+C, lights reset to a neutral white.
#
# Requirements:
#   python3 -m pip install sounddevice numpy requests
#   Have a Hue Bridge on the same network. First run will ask you to press the link button.
#
# Optional files (auto-created):
#   hue_bridge_ip.txt  (cached IP discovered via your existing discover_hue.py)
#   hue_api_key.txt    (local Hue username once paired)
#   hue_group_id.txt   (Room/Zone; defaults to All lights = 0)

import os, time, math, requests, numpy as np, sounddevice as sd
from discover_hue import find_hue_bridge  # uses your existing discovery helper

# ---------------------- User-tweakable settings ----------------------
SAMPLE_RATE   = 48000     # audio sampling rate
CHUNK_SEC     = 1.0       # evaluate once per second (show-first, not ultra fast)
DB_WINDOW     = 20.0      # dB range mapped to brightness 0..1
CAL_OFFSET_DB = 60.0      # shift dBFS into usable range
DELTA_DB_TRIG = 2.0       # how much louder/quieter than last second to change color

# Console + cue toggles
PRINT_DB      = True    # print dB and delta each second
BEEP_ENABLE   = True    # play a short beep indicating up/down change
BEEP_UP_FREQ  = 900.0   # Hz (louder than last second)
BEEP_DOWN_FREQ= 300.0   # Hz (quieter than last second)
BEEP_MS       = 120
BEEP_GAIN     = 0.12
BEEP_COOLDOWN = 0.40    # seconds between beeps to avoid chatter

# Color endpoints (Hue API hue values)
HUE_RED  = 0
HUE_BLUE = 46920
SAT      = 254            # keep colors vivid
BRI_MIN, BRI_MAX = 25, 254

# Transition: 4 = 400 ms (Hue uses 1/10th seconds)
TRANSITION_TENTHS = 4

# Group / Lights
USE_GROUP = True
DEFAULT_GROUP_ID = "0"     # 0 = "All reachable"
API_KEY_FILE   = "hue_api_key.txt"
GROUP_ID_FILE  = "hue_group_id.txt"
BRIDGE_IP_FILE = "hue_bridge_ip.txt"
INPUT_DEV_FILE  = "hue_input_device.txt"
OUTPUT_DEV_FILE = "hue_output_device.txt"

# ---------------------- ASCII ear banner ----------------------
EAR_ASCII = """
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣤⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢠⣾⣿⣿⡿⠋⣁⣤⣤⣤⣌⡙⢿⣿⣷⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢀⣿⣿⠏⣠⣾⠿⠟⠻⢿⣿⣿⣶⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣸⣿⡏⠀⠋⣠⣴⣶⣶⣤⣈⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣿⣿⠀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢻⣿⠀⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⢸⣿⣷⣶⡄⢹⣿⣿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠘⣿⣿⡿⠃⣸⣿⣿⣿⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⣻⣷⣾⣿⣿⣿⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""

# --------------------------------------------------------------------

def get_bridge_ip():
    if os.path.exists(BRIDGE_IP_FILE):
        ip = open(BRIDGE_IP_FILE).read().strip()
        if ip:
            return ip
    ip = find_hue_bridge()
    if ip:
        open(BRIDGE_IP_FILE, "w").write(ip)
    return ip

def load_api_key():
    return open(API_KEY_FILE).read().strip() if os.path.exists(API_KEY_FILE) else ""

def save_api_key(key):
    open(API_KEY_FILE, "w").write(key)

def pair_with_bridge(bridge_ip):
    print("Pairing with Hue Bridge…")
    print("  1) Press the round link button on the Bridge")
    input("  2) Press Enter here within 30 seconds… ")
    try:
        r = requests.post(f"http://{bridge_ip}/api", json={"devicetype":"soundreactive#python"}, timeout=5)
        data = r.json()
        if isinstance(data, list) and data and "success" in data[0]:
            key = data[0]["success"]["username"]
            save_api_key(key)
            print("✅ Paired. API key saved.")
            return key
        print("Unexpected response:", data)
    except Exception as e:
        print("Pairing failed:", e)
    return ""

# --------------- Audio helpers ---------------

def dbfs_from_block(block: np.ndarray) -> float:
    if block.size == 0:
        return -120.0
    # RMS → dBFS
    rms = max(1e-12, float(np.sqrt(np.mean(np.square(block), dtype=np.float64))))
    return 20.0 * math.log10(rms)

# --------------- Mapping helpers ---------------

def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

# hue circle lerp (shortest path)
def hue_lerp(h1, h2, t):
    h1 %= 65536; h2 %= 65536
    d = (h2 - h1) % 65536
    if d > 32768:
        d -= 65536
    return int((h1 + d * clamp01(t)) % 65536)

# brightness mapping from absolute loudness
def t_to_bri(t):
    return int(BRI_MIN + (BRI_MAX - BRI_MIN) * clamp01(t))

# --------------- Beep helper (simple, non-blocking) ---------------
# Uses default output device. If you need a specific output device, we can add a picker.
def play_beep(freq_hz: float, ms: int = BEEP_MS, gain: float = BEEP_GAIN):
    if not BEEP_ENABLE:
        return
    try:
        dur = ms / 1000.0
        n = int(SAMPLE_RATE * dur)
        t = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float32)
        # simple Hann fade in/out to avoid clicks
        env = np.ones(n, dtype=np.float32)
        e = max(16, n // 20)
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, e, dtype=np.float32))
        env[:e] *= ramp
        env[-e:] *= ramp[::-1]
        wave = (gain * np.sin(2.0 * math.pi * freq_hz * t) * env).astype(np.float32)
        # stereo if possible, else mono
        try:
            sd.play(np.column_stack([wave, wave]), samplerate=SAMPLE_RATE, blocking=False)
        except Exception:
            sd.play(wave, samplerate=SAMPLE_RATE, blocking=False)
    except Exception:
        pass

# --------------- Device pickers (input/output) ---------------
def load_input_device():
    if os.path.exists(INPUT_DEV_FILE):
        s = open(INPUT_DEV_FILE).read().strip()
        if s.isdigit():
            return int(s)
    return None

def save_input_device(idx: int):
    open(INPUT_DEV_FILE, "w").write(str(idx))

def choose_input_device_interactive():
    try:
        devs = sd.query_devices()
    except Exception as e:
        print("Could not query audio devices:", e)
        return None
    inputs = [(i, d) for i, d in enumerate(devs) if int(d.get('max_input_channels', 0)) > 0]
    if not inputs:
        print("No input devices found. Using default.")
        return None
    print("\nSelect an input device (microphone):\n")
    for n, (idx, d) in enumerate(inputs, start=1):
        sr = int(d.get('default_samplerate', 48000))
        name = d.get('name', '?')
        print(f"  {n:2d}) idx {idx:2d} | {name} | max_in={d.get('max_input_channels')} | default_sr={sr}")
    sel = input("\nEnter number (default 1): ").strip() or "1"
    if sel.isdigit() and 1 <= int(sel) <= len(inputs):
        return inputs[int(sel)-1][0]
    print("Invalid selection. Using default.")
    return None

def load_output_device():
    if os.path.exists(OUTPUT_DEV_FILE):
        s = open(OUTPUT_DEV_FILE).read().strip()
        if s.isdigit():
            return int(s)
    return None

def save_output_device(idx: int):
    open(OUTPUT_DEV_FILE, "w").write(str(idx))

def choose_output_device_interactive():
    try:
        devs = sd.query_devices()
    except Exception as e:
        print("Could not query audio devices:", e)
        return None
    outputs = [(i, d) for i, d in enumerate(devs) if int(d.get('max_output_channels', 0)) > 0]
    if not outputs:
        print("No output devices found. Using default.")
        return None
    print("\nSelect an output device (speakers/headphones):\n")
    for n, (idx, d) in enumerate(outputs, start=1):
        sr = int(d.get('default_samplerate', 48000))
        name = d.get('name', '?')
        print(f"  {n:2d}) idx {idx:2d} | {name} | max_out={d.get('max_output_channels')} | default_sr={sr}")
    sel = input("\nEnter number (default 1): ").strip() or "1"
    if sel.isdigit() and 1 <= int(sel) <= len(outputs):
        return outputs[int(sel)-1][0]
    print("Invalid selection. Using default.")
    return None

# --------------- Hue REST ---------------

def hue_put(auth_base, path, payload):
    try:
        return requests.put(auth_base + path, json=payload, timeout=3)
    except requests.RequestException:
        return None

# --------------- Main ---------------

def main():
    # Bridge
    bridge_ip = get_bridge_ip()
    if not bridge_ip:
        print("❌ No Hue Bridge found on the network.")
        return
    print(f"✅ Using Hue Bridge at {bridge_ip}")

    # API key
    key = load_api_key() or pair_with_bridge(bridge_ip)
    if not key:
        print("❌ Couldn't obtain an API key.")
        return
    auth = f"http://{bridge_ip}/api/{key}"

    # Group
    gid = open(GROUP_ID_FILE).read().strip() if os.path.exists(GROUP_ID_FILE) else DEFAULT_GROUP_ID
    if not os.path.exists(GROUP_ID_FILE):
        open(GROUP_ID_FILE, "w").write(gid)
    target_path = f"/groups/{gid}/action" if USE_GROUP else "/lights/1/state"
    print(f"Controlling group id: {gid}")

    # ---- Input / Output device selection ----
    in_idx = load_input_device()
    if in_idx is None:
        in_idx = choose_input_device_interactive()
        if in_idx is not None:
            save_input_device(in_idx)
    # Validate input device
    if in_idx is not None:
        try:
            info = sd.query_devices(in_idx)
            if int(info.get('max_input_channels', 0)) < 1:
                print("Saved input device has 0 input channels. Using default.")
                in_idx = None
            else:
                print(f"Using input device {in_idx}: {info.get('name','?')}")
        except Exception:
            print("Saved input device not available. Using default.")
            in_idx = None
    else:
        print("Using default input device.")

    out_idx = load_output_device()
    if out_idx is None:
        out_idx = choose_output_device_interactive()
        if out_idx is not None:
            save_output_device(out_idx)
    # Validate output device
    if out_idx is not None:
        try:
            info = sd.query_devices(out_idx)
            if int(info.get('max_output_channels', 0)) < 1:
                print("Saved output device has 0 output channels. Using default.")
                out_idx = None
            else:
                print(f"Using output device {out_idx}: {info.get('name','?')}")
        except Exception:
            print("Saved output device not available. Using default.")
            out_idx = None
    else:
        print("Using default output device.")

    # Apply defaults so sd.play() uses the selected output, and InputStream uses selected input
    try:
        sd.default.device = (in_idx, out_idx)
    except Exception:
        # Fallback: ignore if tuple invalid; we'll pass device explicitly to InputStream
        pass

    # Audio stream
    frames_per_chunk = int(SAMPLE_RATE * CHUNK_SEC)
    last_db = None
    color_t = 0.5  # 0=blue, 1=red; start neutral-ish
    last_beep_ts = 0.0

    print("\nStarting mic (1-second windows). Ctrl+C to stop.\n")
    try:
        print("\n" + EAR_ASCII + "\n   ( listening… )\n")
    except Exception:
        pass

    try:
        with sd.InputStream(device=in_idx,  # may be None → default
                            channels=1, samplerate=SAMPLE_RATE,
                            blocksize=frames_per_chunk, dtype="float32",
                            latency=0.03) as stream:
            while True:
                # Read exactly 1 second of audio
                buf, _ = stream.read(frames_per_chunk)
                mono = buf[:, 0].astype(np.float32)

                # Loudness for this window
                db = dbfs_from_block(mono) + CAL_OFFSET_DB
                db = max(0.0, db)   # floor at 0 dB effective

                # Brightness from absolute level
                t_bri = clamp01(db / DB_WINDOW)  # 0..1 across DB_WINDOW range
                bri   = t_to_bri(t_bri)

                # Color from trend vs previous window
                if last_db is None:
                    delta = 0.0
                else:
                    delta = db - last_db

                if delta > DELTA_DB_TRIG:
                    # Louder → warm/red side
                    color_t = min(1.0, color_t + 0.25)
                elif delta < -DELTA_DB_TRIG:
                    # Quieter → cool/blue side
                    color_t = max(0.0, color_t - 0.25)
                else:
                    # Small change → drift gently toward middle (pleasant stability)
                    color_t = color_t * 0.9 + 0.5 * 0.1

                hue = hue_lerp(HUE_BLUE, HUE_RED, color_t)

                # One smooth Hue update per second
                payload = {"on": True, "bri": bri, "hue": hue, "sat": SAT, "transitiontime": TRANSITION_TENTHS}
                hue_put(auth, target_path, payload)

                # Optional console print
                if PRINT_DB:
                    print(f"dB={db:.1f}  Δ={delta:+.1f}  bri={bri:3d}  hue={hue}")

                # Optional up/down beeps with cooldown
                now_ts = time.time()
                if delta > DELTA_DB_TRIG and (now_ts - last_beep_ts) >= BEEP_COOLDOWN:
                    play_beep(BEEP_UP_FREQ)
                    last_beep_ts = now_ts
                elif delta < -DELTA_DB_TRIG and (now_ts - last_beep_ts) >= BEEP_COOLDOWN:
                    play_beep(BEEP_DOWN_FREQ)
                    last_beep_ts = now_ts

                # For next pass
                last_db = db

    except KeyboardInterrupt:
        pass
    finally:
        # Reset to neutral white on exit
        try:
            reset = {"on": True, "ct": 366, "bri": 200, "transitiontime": 4}
            hue_put(auth, target_path, reset)
            time.sleep(0.2)
            print("\nLights reset to white. Bye!")
        except Exception:
            pass


if __name__ == "__main__":
    main()