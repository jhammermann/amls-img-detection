AMLS AI Image Detection
=======================

The data folder is read-only (requirement from the task pdf) and as a reminder, I named it like that. We'll change the name to 'data' later.

Installation: 
```bash
cd solution/

sudo apt update && sudo apt install -y python3.11-venv && python3.11 -m venv .venv && source .venv/bin/activate

pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
pip install -r requirements.txt

mkdir artifacts && mkdir data-readonly
```
Then, paste the downloaded data into data-readonly.