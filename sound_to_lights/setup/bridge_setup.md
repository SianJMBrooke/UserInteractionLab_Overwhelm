Here’s the plain text for setup/bridge_setup.md

# Hue Bridge Setup

This guide explains how to set up and pair the Philips Hue Bridge for the Overhelm exhibit.


## Hardware Setup

1. Connect the Hue Bridge to power (white LEDs on the top should light up).  
2. Connect the Hue Bridge to the exhibit network via **Ethernet cable** (see [Network Instructions](network_instructions.md)).  
3. Wait until the middle LED (network symbol) is **solid**.  
   - Flashing = not connected  
   - Off = cable issue  
   - Solid = ready  

## Pairing with the Script

1. Run the script:  
   ```bash
   python3 hue_sound_reactive.py
```
	1.	When prompted, press the round link button on the Hue Bridge.
	2.	The script will generate and save an API key in hue_api_key.txt.
	3.	Only needed once — subsequent runs reuse this key.


## Verifying the Setup
	•	Check in the Philips Hue app that all bulbs are connected and “reachable.”
	•	Run the script and ensure that group/room selection lists the bulbs.
	•	Make sound → bulbs should change color and brightness.
