
# Network Setup – Overhelm Exhibit

The Philips Hue Bridge requires a **wired Ethernet connection** to function.  
Because networks like **Eduroam** or the **Visualisation Lab WiFi** do not allow direct device discovery, we use a **TP-Link AC1750 WiFi Range Extender (RE450)** in Access Point (AP) mode.


## Why Use the TP-Link RE450?

- Eduroam blocks the discovery protocols used by the Hue Bridge.  
- The Hue Bridge cannot connect directly to campus WiFi networks.  
- The RE450 creates a **private local network** that the MacBook and Hue Bridge can share.  
- This ensures the exhibit computer and Hue Bridge are on the same subnet.  


## Hardware Used

- **TP-Link AC1750 WiFi Range Extender (RE450)**  
  - 1× Gigabit LAN port (for the Hue Bridge)  
  - 3 fixed antennas  
  - Modes: Extender / Access Point  


## Setup Steps

### 1. Configure the Extender
1. Plug in the TP-Link RE450 near the exhibit.  
2. Press the **Reset button** (if reusing from another setup).  
3. Connect to the extender via WiFi from your MacBook (network will be called `TP-LINK_Extender_XXXX`).  
4. Open a browser and go to `http://tplinkrepeater.net` (or the IP address printed on the label).  
5. Login (default: `admin` / `admin` if not configured).  
6. Switch the RE450 to **Access Point (AP Mode)**.  

### 2. Connect the Extender to Eduroam / Lab WiFi
1. In the setup wizard, select **Eduroam** or **Visualisation Lab WiFi** as the upstream WiFi.  
   - If Eduroam is not supported directly (due to enterprise authentication), connect the RE450 to the **Visualisation Lab WiFi** instead.  
2. Set a **new private SSID** for the local network (e.g., `OverhelmNet`).  
3. Save and reboot the extender.  

### 3. Connect the Hue Bridge
- Plug an Ethernet cable from the **LAN port of the RE450** into the **Hue Bridge**.
- Wait until the middle LED (network icon) on the Hue Bridge is **solid**.  

### 4. Connect the Exhibit MacBook
- Connect the MacBook’s WiFi to the private SSID you set on the RE450 (e.g., `OverhelmNet`).
- Confirm the MacBook has an IP address in the same range as the Bridge (e.g., `192.168.0.x`).
- Check in Terminal:  
     ```bash
     ifconfig en0
     ```  


## Verification

- Run in Terminal:  
  ```bash
  arp -a
  ```
You should see the Hue Bridge listed with an IP like 192.168.0.101.

Run the script:

``` bash
python3 hue_sound_reactive.py
```
It should auto-discover the Bridge and prompt for pairing.


## Alternative: Internet Sharing from Mac

If the RE450 is unavailable:
	-	Connect the MacBook to Eduroam or Lab WiFi.
	-	Enable Internet Sharing in macOS:
	-	Go to System Settings → Sharing → Internet Sharing
	-	Share connection: Wi-Fi → Ethernet
	-	Connect the Hue Bridge to the Mac via Ethernet adapter.
	-	The Mac will assign the Bridge an IP (usually 192.168.2.2).


### Notes:
	-	The RE450 must stay powered on throughout the exhibit.
	-	Always verify the Hue Bridge LEDs:
	-	Left = Power
	-	Middle = Network (must be solid)
	-	Right = Internet


If the middle LED is flashing or off, check cabling or reset the RE450.
