---
name: youtube-transcript
description: Extract transcripts from YouTube videos. Use when user shares a YouTube link and wants the transcript, captions, or subtitles.
---

# YouTube Transcript Skill

Extract transcripts from YouTube videos using yt-dlp. Supports manual subtitles, auto-generated captions, language selection, and multiple output formats.

## Quick Usage

```bash
# Get transcript (auto-detect best available)
.venv/jarvis/bin/python .claude/skills/youtube-transcript/scripts/yt_transcript.py "https://youtube.com/watch?v=VIDEO_ID"

# Specify language
.venv/jarvis/bin/python .claude/skills/youtube-transcript/scripts/yt_transcript.py "URL" --lang en

# List available subtitle languages
.venv/jarvis/bin/python .claude/skills/youtube-transcript/scripts/yt_transcript.py "URL" --list-subs

# Output as JSON (with timestamps)
.venv/jarvis/bin/python .claude/skills/youtube-transcript/scripts/yt_transcript.py "URL" --json

# Include timestamps in plaintext
.venv/jarvis/bin/python .claude/skills/youtube-transcript/scripts/yt_transcript.py "URL" --with-timestamps

# Save to file
.venv/jarvis/bin/python .claude/skills/youtube-transcript/scripts/yt_transcript.py "URL" -o output.txt
```

## Subtitle Priority

1. **Manual subtitles** (human-written) — preferred when available
2. **Auto-generated captions** — fallback
3. Error with clear message if neither available

## Supported URL Formats

All standard YouTube URL formats are supported:
- `https://www.youtube.com/watch?v=ID`
- `https://youtu.be/ID`
- `https://m.youtube.com/watch?v=ID`
- `https://youtube.com/embed/ID`

## Dependencies

- `yt-dlp` (installed in `.venv/jarvis/`)
