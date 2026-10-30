# Virality Simulator (prototype)

Reel upload -> AI personas react -> virality potential estimate.
Ye abhi Phase 1 hai: key test + reel se frames aur transcript nikalna.

## Setup

1. Python 3.10+ aur ffmpeg install karo
   - Windows: `winget install ffmpeg`
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
2. Virtual environment aur packages:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt
   ```
3. `.env.example` ko copy karke `.env` banao aur apni key daalo:
   ```
   GEMINI_API_KEY=tumhari_key
   GEMINI_MODEL=gemini-3.8-flash
   ```
   `.env` ko kabhi GitHub pe mat daalna (`.gitignore` mein already hai).

## Phase 1 ke test

```
python test_key.py                      # key chal rahi hai ya nahi
python -m core.extract path/to/reel.mp4 # frames + transcript
```

Frames `tmp/frames/` mein save hote hain.
