# Car Rental Scraper —  Setup Guide

This tool collects car prices from **Sixt** and **Enterprise** websites and saves them to an **Excel-friendly CSV file**.

No coding needed — just follow the steps below and copy-paste the commands.

---

## What you need before starting

Install these **once** on your computer:

| What to install | Where to get it | Why |
|-----------------|-----------------|-----|
| **Python 3.12** (recommended) | [python.org/downloads](https://www.python.org/downloads/) | Runs the scraper |
| **VS Code** (optional but helpful) | [code.visualstudio.com](https://code.visualstudio.com/) | Easy way to open the project |
| **Ollama** (for best results) | [ollama.com](https://ollama.com) | AI that reads the car listings |

> **Windows tip:** When installing Python, tick the box **"Add Python to PATH"**.

---

## Step-by-step setup (first time only)

Follow every step in order. Do not skip steps.

---

### STEP 1 — Get the project folder on your computer

**If someone sent you a ZIP file:**
1. Unzip it
2. You should see a folder called `car-rental-scraper`

**If it's on GitHub:**
1. Click the green **Code** button → **Download ZIP**
2. Unzip it

**Open it in VS Code (recommended):**
1. Open VS Code
2. Click **File → Open Folder**
3. Select the `car-rental-scraper` folder
4. Click **Open**

---

### STEP 2 — Open the terminal

In VS Code:
1. Click **Terminal** in the top menu
2. Click **New Terminal**

A black/white box appears at the bottom — this is where you type commands.

Make sure you are in the right folder. You should see `car-rental-scraper` in the terminal path.

---

### STEP 3 — Create a virtual environment

Copy and paste this **one line at a time** (Mac):

```bash
python3 -m venv venv
```

Then:

```bash
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**How do you know it worked?**  
You should see `(venv)` at the start of your terminal line, like:

```
(venv) om_patel@macbookpro car-rental-scraper %
```

> You only create `venv` once. But **every time you open a new terminal**, run `source venv/bin/activate` again before scraping.

---

### STEP 4 — Install Python packages

Copy and paste:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Wait until it finishes. You should see **"Successfully installed"** at the end.

**What gets installed:**

| Package | What it does |
|---------|--------------|
| `pandas` | Creates the CSV file |
| `playwright` | Opens the website in a browser |
| `requests` | Connects to the AI |
| `pyyaml` | Config support |

**If you see an error about `greenlet` or build failed:**  
You may have Python 3.14. Use Python 3.12 instead:

```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### STEP 5 — Install the browser (IMPORTANT — do not skip)

After Step 4, you **must** run this:

```bash
playwright install chromium
```

This downloads Chrome for the scraper. It takes 1–2 minutes.

**If you skip this step**, you will get this error when running:

```
Executable doesn't exist at .../ms-playwright/chromium-...
Please run: playwright install
```

**Fix:** Just run:

```bash
playwright install chromium
```

Run this again any time you update packages with `pip install`.

---

### STEP 6 — Install Ollama (for AI mode — recommended)

1. Go to [ollama.com](https://ollama.com) and install Ollama
2. Open a **new terminal tab** (Terminal → New Terminal in VS Code)
3. Run:

```bash
ollama serve
```

Leave this terminal open.

4. Open **another new terminal tab**, activate venv, and download the AI models:

```bash
source venv/bin/activate
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

This takes several minutes the first time (large download).

> **Don't want AI?** You can use `--hybrid` instead of `--ai-merge` and skip Ollama. Results may be less complete.

---

## How to run the scraper (every time)

Every time you want to scrape, do this:

### 1. Open VS Code → open the `car-rental-scraper` folder

### 2. Open terminal and activate venv

**Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```cmd
venv\Scripts\activate
```

### 3. Make sure Ollama is running (if using `--ai-merge`)

In a separate terminal tab:
```bash
ollama serve
```

### 4. Run one of these commands

**Enterprise — Toronto Airport:**
```bash
python3 run.py --site enterprise --location "Toronto Airport" --ai-merge
```

**Sixt — Calgary:**
```bash
python3 run.py --site sixt --location "YYC" --ai-merge
```

**Sixt — Vancouver:**
```bash
python3 run.py --site sixt --location "YVR" --ai-merge
```

A browser window will open, search the site, and collect car prices. **Do not close the browser** — the scraper closes it automatically.

---

## Where to find your results

After a successful run, open this folder:

```
car-rental-scraper/scraper_outputs/
```

Example path:
```
scraper_outputs/Enterprise_Car_Rental/Toronto_Airport/20260618_121242.csv
```

**Open the `.csv` file in Excel or Google Sheets.**

Each run creates 4 files:

| File | What it is |
|------|------------|
| `.csv` | **Your data** — open in Excel |
| `.json` | Same data in JSON format |
| `.log` | Step-by-step log of what happened |
| `.py` | Replay script (ignore unless you are a developer) |

### Columns in the CSV

| Column | Example |
|--------|---------|
| pickup_date | 2026-06-19 |
| return_date | 2026-06-20 |
| pickup_time | 10:00 |
| return_time | 10:00 |
| car_name | Nissan Kicks or similar |
| car_type | Compact SUV |
| price_per_day | CA$61.78/day |
| transmission | Automatic |
| seats | 5 |
| bags | 3 |
| location | Toronto Airport |

---

## Quick reference — copy-paste commands

**Full first-time setup (Mac):**
```bash
cd car-rental-scraper
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
playwright install chromium
ollama pull qwen2.5:7b
```

**Every time you scrape (Mac):**
```bash
source venv/bin/activate
python3 run.py --site enterprise --location "Toronto Airport" --ai-merge
```

---

## Common problems and fixes

### "No module named 'requests'" or "No module named 'pandas'"

Packages are not installed. Run:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

---

### "Executable doesn't exist at .../ms-playwright/chromium-..."

Browser not downloaded. Run:
```bash
playwright install chromium
```

---

### "Failed building wheel for greenlet"

Python version too new. Use Python 3.12:
```bash
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

---

### `(venv)` does not show in terminal

Virtual environment is not active. Run:
```bash
source venv/bin/activate
```

---

### Scraper hangs or AI errors

Ollama is not running. Open a new terminal and run:
```bash
ollama serve
```

---

### 0 vehicles found

Try a different mode:
```bash
python3 run.py --site enterprise --location "Toronto Airport" --hybrid
```

Or run **without** `--headless` so you can see if a pop-up is blocking the page.

---

## What each file in the project does

| File | Do you need to edit it? |
|------|-------------------------|
| `run.py` | No — this is what you run |
| `agent_core.py` | No — the scraper engine |
| `requirements.txt` | No — list of packages to install |
| `README.md` | No — this guide |
| `.gitignore` | No — for GitHub |

---

## requirements.txt (for reference)

```
pandas>=2.2.0
playwright>=1.49.0
pyyaml>=6.0.1
requests>=2.31.0
```

---

## Example — what success looks like

```
============================================================
  Universal Car-Rental Scraper
  Site     : https://www.enterprise.ca
  Location : Toronto Airport
  AI mode  : AI-MERGE (Ollama first + parser/DOM merge)
============================================================
...
Done. Found 41 vehicles.

==========================================================
  41 records
  scraper_outputs/Enterprise_Car_Rental/Toronto_Airport
     csv      -> 20260618_121242.csv
==========================================================
```

Open that CSV file — you will see 41 rows of car data.

---
## License

Use at your own risk. Respect each website's terms of service.
