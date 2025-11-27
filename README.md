# Instruction

## Setup
```bash
git clone https://github.com/Luthen-EchoBot/gestures_detection.git
cd gestures_detection
pyenv install 3.10.14
pyenv shell 3.10.14
python -m venv venv
source venv/bin/activate
pip install mediapipe
```

## Use
```bash
source venv/bin/activate # skip if terminal starts with "(venv)"
python gesture_console.py
```

## Use
# Connecter a la jetson
```bash
ssh pi@10.105.1.167
```
Password: geicar
```bash
ssh jetson@192.168.1.10
```
Password: jetson
```bash
cd AI/GestureDetecion/gestures_detection
source venv/bin/activate
```
