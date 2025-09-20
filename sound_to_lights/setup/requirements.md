
# Requirements – Overhelm Exhibit: Sound/Light

This document explains how to install the required software for the **sound-reactive Hue lights** used in the Overhelm exhibit.

---

## System Requirements

- macOS (tested on MacBook Pro, macOS 14+)  
- Python 3.9 or higher  
- Internet access for package installation  
- Philips Hue Bridge and bulbs (already paired in Hue app)  

---

## Python Setup

### 1. Check Python installation
Open **Terminal** and type:
```bash
python3 --version
```

You should see something like Python 3.9.6 or newer.
If not installed, download Python from python.org/downloads.


### 2. Install pip (if missing)

On most macOS systems pip is already installed. If not, run:
``` bash
python3 -m ensurepip --upgrade
```
Then confirm with:
``` bash
python3 -m pip --version
```

### 3. Install project dependencies

From the project root folder, run:
``` bash
python3 -m pip install -r requirements.txt
```
This installs:
- sounddevice – microphone input
- numpy – audio signal processing
- requests – communication with the Hue Bridge


## First Run

After installing requirements, run the program:
``` bash
python3 hue_sound_reactive.py
```

On the first run, the script will:
	1.	Ask you to press the link button on the Hue Bridge
	2.	Prompt you to select which Room/Zone to control
	3.	Prompt you to select a microphone input (Built-in Microphone recommended)

Settings are saved locally for future runs.

## Resetting Saved Settings

If you need to re-pair or reconfigure, delete the local files:

``` bash
rm hue_api_key.txt hue_group_id.txt hue_input_device.txt
```

The script will prompt you again on the next run.
