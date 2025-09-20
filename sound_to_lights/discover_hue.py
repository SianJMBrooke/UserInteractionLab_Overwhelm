import http.client

def find_hue_bridge(subnet="192.168.1"):
    def is_hue(ip):
        try:
            conn = http.client.HTTPConnection(ip, 80, timeout=0.2)
            conn.request("GET", "/api/config")
            res = conn.getresponse()
            if res.status == 200:
                data = res.read().decode("utf-8")
                return '"bridgeid"' in data.lower() or '"philips hue"' in data.lower()
        except:
            return False
        return False

    for i in range(2, 255):
        ip = f"{subnet}.{i}"
        if is_hue(ip):
            return ip
    return None

if __name__ == "__main__":
    ip = find_hue_bridge()
    if ip:
        print(f"✅ Found Hue Bridge at: {ip}")
    else:
        print("❌ No Hue Bridge found")