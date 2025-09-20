# Hue sound-reactive lights — fast reaction version
# - First run: press the Bridge link button when prompted; key saved to hue_api_key.txt.
# - First run: pick a Room/Zone and an input device; saved to hue_group_id.txt / hue_input_device.txt.
# - Install deps:
#     python3 -m pip install sounddevice numpy requests

import time, math, threading, queue, requests, numpy as np, sounddevice as sd, http.client, os

# ================= USER SETTINGS (tuned for quick reaction) =================
BRIDGE_IP = None                 # e.g. "192.168.2.2"; leave None to auto-discover
API_KEY_FILE   = "hue_api_key.txt"
GROUP_ID_FILE  = "hue_group_id.txt"
INPUT_DEV_FILE = "hue_input_device.txt"

USE_GROUP = True                 # keep True; chosen group id is stored in GROUP_ID_FILE
GROUP_ID  = "0"                  # fallback if no saved choice; "0" = all reachable lights
LIGHT_IDS = ["1","2"]            # only used if USE_GROUP=False

# Audio (lower block size → lower latency)
SAMPLE_RATE  = 48000
BLOCK_SIZE   = 256               # was 1024; smaller = quicker response

# Sensitivity / envelopes (faster attack, slightly quicker floor)
VU_SMOOTHING    = 0.35           # was 0.7; lower = faster
FLOOR_SMOOTHING = 0.94           # was 0.97; tracks room floor a bit quicker
DB_WINDOW       = 18.0           # dB range from quiet→loud mapping
CAL_OFFSET_DB   = 60.0           # shifts dBFS into usable range
MIN_EFF_DB      = 20.0

# Output pacing (more updates; smaller deltas trigger sends)
UPDATE_HZ     = 20               # was 12
MIN_DELTA_BRI = 2                # was 3
MIN_DELTA_HUE = 800              # was 1500
SATURATION    = 254
BRI_MIN, BRI_MAX = 25, 254
# ===========================================================================

def discover_bridge_ip():
    subnets = ["192.168.2.", "192.168.137.", "10.42.0.", "192.168.0.", "192.168.1.", "10.0.0."]
    def is_hue(ip):
        try:
            c = http.client.HTTPConnection(ip, 80, timeout=0.25)
            c.request("GET", "/api/config")
            r = c.getresponse()
            if r.status == 200:
                body = r.read().decode("utf-8","ignore").lower()
                return '"bridgeid"' in body or '"philips hue"' in body
        except Exception:
            return False
        return False
    for base in subnets:
        for last in range(2,255):
            ip = f"{base}{last}"
            if is_hue(ip):
                print(f"Found Hue Bridge at {ip}")
                return ip
    return None

# --- API key storage ---
def load_api_key():
    if os.path.exists(API_KEY_FILE):
        k = open(API_KEY_FILE).read().strip()
        if k:
            print("Loaded API key from file.")
            return k
    return ""

def save_api_key(k):
    open(API_KEY_FILE,"w").write(k)
    print("Saved API key to", API_KEY_FILE)

def pair_get_api_key(bridge_ip):
    print("Pairing with Hue Bridge:")
    print("  1) Press the round link button on the Bridge")
    input("  2) Press Enter here within 30 seconds... ")
    try:
        r = requests.post(f"http://{bridge_ip}/api", json={"devicetype":"sound_reactive#python"}, timeout=4)
        data = r.json()
        if isinstance(data,list) and data and "success" in data[0] and "username" in data[0]["success"]:
            key = data[0]["success"]["username"]
            print("Got API key:", key)
            save_api_key(key)
            return key
        print("Unexpected response:", data)
        return ""
    except Exception as e:
        print("Pairing failed:", e)
        return ""

# --- Group selection ---
def load_group_id():
    if os.path.exists(GROUP_ID_FILE):
        gid = open(GROUP_ID_FILE).read().strip()
        if gid:
            print(f"Loaded group id {gid} from file.")
            return gid
    return ""

def save_group_id(gid):
    open(GROUP_ID_FILE,"w").write(gid)
    print("Saved group id", gid, "to", GROUP_ID_FILE)

def list_groups(auth_base):
    out = [("0","All lights","Special")]
    try:
        res = requests.get(f"{auth_base}/groups", timeout=4)
        groups = res.json()
        if isinstance(groups, dict):
            for gid, g in groups.items():
                name = g.get("name","")
                gtype = g.get("type","")
                if gtype in ("Room","Zone"):
                    out.append((gid, name, gtype))
    except Exception:
        pass
    return out

def choose_group_interactive(auth_base):
    groups = list_groups(auth_base)
    if not groups:
        print("No groups found; defaulting to 0 (All lights).")
        return "0"
    print("\nSelect a group to control:\n")
    for i,(gid,name,gtype) in enumerate(groups, start=1):
        print(f"  {i:2d}) {name}  [{gtype}]  (id: {gid})")
    while True:
        sel = input("\nEnter number (default 1 for All lights): ").strip() or "1"
        if sel.isdigit() and 1 <= int(sel) <= len(groups):
            chosen = groups[int(sel)-1][0]
            print(f"Selected group id {chosen} ({groups[int(sel)-1][1]}).")
            return chosen
        print("Invalid selection. Try again.")

# --- Input device ---
def load_input_device():
    if os.path.exists(INPUT_DEV_FILE):
        s = open(INPUT_DEV_FILE).read().strip()
        if s.isdigit():
            idx = int(s)
            print(f"Loaded input device index {idx} from file.")
            return idx
    return None

def save_input_device(idx):
    open(INPUT_DEV_FILE, "w").write(str(idx))
    print("Saved input device index", idx, "to", INPUT_DEV_FILE)

def choose_input_device_interactive():
    try:
        devs = sd.query_devices()
    except Exception as e:
        print("Could not query devices:", e)
        return None
    inputs = [(i,d) for i,d in enumerate(devs) if d.get("max_input_channels",0) > 0]
    if not inputs:
        print("No input devices found. Using default device.")
        return None
    print("\nSelect an input device (microphone):\n")
    for i,(idx,d) in enumerate(inputs, start=1):
        sr = int(d.get("default_samplerate", 48000))
        print(f"  {i:2d}) idx {idx:2d} | {d.get('name','?')} | max_in={d.get('max_input_channels')} | default_sr={sr}")
    # Prefer Built-in Microphone if present
    default_choice = next((i for i,(idx,d) in enumerate(inputs, start=1)
                           if "built-in microphone" in d.get("name","").lower()), 1)
    sel = input(f"\nEnter number (default {default_choice}): ").strip() or str(default_choice)
    if sel.isdigit() and 1 <= int(sel) <= len(inputs):
        idx = inputs[int(sel)-1][0]
        print(f"Selected input device index {idx}: {inputs[int(sel)-1][1].get('name','?')}")
        return idx
    print("Invalid selection. Using default device.")
    return None

# --- Audio & Hue mapping ---
def dbfs_from_block(block: np.ndarray) -> float:
    if block.size == 0: return -120.0
    rms = max(1e-12, float(np.sqrt(np.mean(np.square(block), dtype=np.float64))))
    return 20.0 * math.log10(rms)

def clamp01(x): return 0.0 if x<0.0 else 1.0 if x>1.0 else x

HUE_RED, HUE_BLUE = 0, 46920
def hue_lerp_circle(h1,h2,t):
    h1%=65536; h2%=65536
    d=(h2-h1)%65536
    if d>32768: d-=65536
    return int((h1 + d*clamp01(t))%65536)
def t_to_hue(t): return hue_lerp_circle(HUE_BLUE, HUE_RED, t)
def t_to_bri(t): return int(BRI_MIN + (BRI_MAX-BRI_MIN)*clamp01(t))

def hue_put(base_url, path, payload):
    try:
        return requests.put(base_url+path, json=payload, timeout=2)
    except requests.RequestException:
        return None

# --- Main ---
def main():
    global BRIDGE_IP, GROUP_ID

    # Bridge IP
    if not BRIDGE_IP:
        print("Discovering Hue Bridge...")
        BRIDGE_IP = discover_bridge_ip()
        if not BRIDGE_IP:
            print("No Hue Bridge discovered. Make sure:")
            print("  • The Bridge is powered on")
            print("  • Ethernet cable is connected and the middle LED is solid")
            print("  • Your Mac and Bridge are reachable on the same network")
            return
    base = f"http://{BRIDGE_IP}/api"
    print("Using Bridge:", BRIDGE_IP)

    # API key
    API_KEY = load_api_key()
    if not API_KEY:
        API_KEY = pair_get_api_key(BRIDGE_IP)
        if not API_KEY:
            print("No API key. Exiting.")
            return
    auth = f"{base}/{API_KEY}"

    # Group
    if USE_GROUP:
        gid_saved = load_group_id()
        if gid_saved:
            GROUP_ID = gid_saved
        else:
            GROUP_ID = choose_group_interactive(auth)
            save_group_id(GROUP_ID)
        print("Controlling group id:", GROUP_ID)

    # Input device
    dev_index = load_input_device()
    if dev_index is None:
        dev_index = choose_input_device_interactive()
        if dev_index is not None:
            save_input_device(dev_index)
    if dev_index is None:
        print("Using default input device (system default).")
    else:
        name = sd.query_devices()[dev_index]["name"]
        print(f"Using input device {dev_index}: {name}")

    # Sender thread (instant transitions, higher rate)
    q_cmd = queue.Queue(maxsize=5)
    last = {"bri": None, "hue": None}
    last_send = 0.0
    send_interval = 1.0 / max(1, UPDATE_HZ)

    def sender():
        nonlocal last, last_send
        while True:
            item = q_cmd.get()
            if item is None: break
            bri, hue = item
            now = time.time()
            send = (
                last["bri"] is None or
                abs(bri-last["bri"])>=MIN_DELTA_BRI or
                abs(hue-last["hue"])>=MIN_DELTA_HUE or
                (now-last_send)>=1.0
            )
            if send and (now-last_send)>=send_interval:
                payload = {"on": True, "bri": bri, "hue": hue, "sat": SATURATION, "transitiontime": 0}  # instant
                if USE_GROUP:
                    hue_put(auth, f"/groups/{GROUP_ID}/action", payload)
                else:
                    for lid in LIGHT_IDS:
                        hue_put(auth, f"/lights/{lid}/state", payload)
                last = {"bri": bri, "hue": hue}; last_send = now

    t_send = threading.Thread(target=sender, daemon=True); t_send.start()

    # Audio processing (fast attack + adaptive floor)
    vu_db, floor_db = None, None

    def audio_cb(indata, frames, time_info, status):
        nonlocal vu_db, floor_db
        d = indata.astype(np.float32)
        if d.ndim == 2: d = np.mean(d, axis=1)
        raw_db = dbfs_from_block(d)
        eff_db = max(MIN_EFF_DB, raw_db + CAL_OFFSET_DB)

        # Fast attack VU
        vu_db = eff_db if vu_db is None else (VU_SMOOTHING*vu_db + (1.0-VU_SMOOTHING)*eff_db)

        # Adaptive floor: quicker to follow quieting, modestly faster when getting louder
        if floor_db is None:
            floor_db = vu_db
        else:
            if vu_db < floor_db:
                floor_db = (FLOOR_SMOOTHING*floor_db + (1.0-FLOOR_SMOOTHING)*vu_db)
            else:
                floor_db = (0.990*floor_db + 0.010*vu_db)

        rel_db = max(0.0, vu_db - floor_db)
        t = clamp01(rel_db / DB_WINDOW)  # 0..1 across selected window

        hue, bri = t_to_hue(t), t_to_bri(t)
        try:
            if q_cmd.full(): q_cmd.get_nowait()
            q_cmd.put_nowait((bri, hue))
        except queue.Full:
            pass

    print("Starting mic. Ctrl+C to stop.")
    try:
        with sd.InputStream(device=dev_index, channels=1, samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE, dtype="float32",
                            latency="low",                # request low-latency path
                            callback=audio_cb):
            while True: time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        q_cmd.put(None); print("Stopped.")

if __name__ == "__main__":
    main()