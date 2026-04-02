#!/usr/bin/env python3
"""Extract transcripts from YouTube videos using yt-dlp."""

import argparse
import json
import re
import sys
import io

import yt_dlp


def list_subtitles(url):
    """List available subtitle languages for a video."""
    opts = {"quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    result = {"title": info.get("title", ""), "manual": {}, "automatic": {}}

    for lang, tracks in (info.get("subtitles") or {}).items():
        result["manual"][lang] = [t.get("ext", "?") for t in tracks]

    for lang, tracks in (info.get("automatic_captions") or {}).items():
        result["automatic"][lang] = [t.get("ext", "?") for t in tracks]

    return result


def extract_transcript(url, lang="en"):
    """Extract transcript from a YouTube video.

    Tries manual subtitles first, then auto-generated captions.
    Returns (segments, metadata) where segments is list of {text, start, end}.
    """
    opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "vtt/srt/best",
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title", "")
    manual_subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    # Try manual first, then auto
    sub_tracks = None
    source = None
    if lang in manual_subs:
        sub_tracks = manual_subs[lang]
        source = "manual"
    elif lang in auto_subs:
        sub_tracks = auto_subs[lang]
        source = "auto-generated"
    else:
        available = list(manual_subs.keys()) + list(auto_subs.keys())
        raise ValueError(
            f"No subtitles found for language '{lang}'. "
            f"Available: {', '.join(sorted(set(available))) or 'none'}"
        )

    # Find VTT or SRT track
    sub_url = None
    sub_ext = None
    for track in sub_tracks:
        ext = track.get("ext", "")
        if ext in ("vtt", "srt"):
            sub_url = track.get("url")
            sub_ext = ext
            break
    if not sub_url and sub_tracks:
        sub_url = sub_tracks[0].get("url")
        sub_ext = sub_tracks[0].get("ext", "vtt")

    if not sub_url:
        raise ValueError("Could not find subtitle download URL.")

    # Download subtitle content
    import urllib.request
    with urllib.request.urlopen(sub_url) as resp:
        content = resp.read().decode("utf-8", errors="replace")

    if sub_ext == "srt" or not sub_ext:
        segments = parse_srt(content)
    else:
        segments = parse_vtt(content)

    metadata = {"title": title, "language": lang, "source": source}
    return segments, metadata


def parse_vtt(vtt_content):
    """Parse WebVTT content into deduplicated segments."""
    lines = vtt_content.split("\n")
    segments = []
    timestamp_re = re.compile(
        r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
    )
    tag_re = re.compile(r"<[^>]+>")

    i = 0
    while i < len(lines):
        match = timestamp_re.search(lines[i])
        if match:
            start = _ts_to_seconds(match.group(1))
            end = _ts_to_seconds(match.group(2))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() and not timestamp_re.search(lines[i]):
                line = tag_re.sub("", lines[i]).strip()
                if line:
                    text_lines.append(line)
                i += 1
            text = " ".join(text_lines)
            if text:
                segments.append({"text": text, "start": start, "end": end})
        else:
            i += 1

    return _deduplicate(segments)


def parse_srt(srt_content):
    """Parse SRT content into deduplicated segments."""
    segments = []
    timestamp_re = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
    )
    tag_re = re.compile(r"<[^>]+>")

    blocks = re.split(r"\n\s*\n", srt_content.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        for j, line in enumerate(lines):
            match = timestamp_re.search(line)
            if match:
                start = _ts_to_seconds(match.group(1).replace(",", "."))
                end = _ts_to_seconds(match.group(2).replace(",", "."))
                text_lines = []
                for tl in lines[j + 1:]:
                    cleaned = tag_re.sub("", tl).strip()
                    if cleaned:
                        text_lines.append(cleaned)
                text = " ".join(text_lines)
                if text:
                    segments.append({"text": text, "start": start, "end": end})
                break

    return _deduplicate(segments)


def _ts_to_seconds(ts):
    """Convert HH:MM:SS.mmm timestamp to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def _deduplicate(segments):
    """Remove duplicate/overlapping text from auto-generated captions.

    YouTube auto-subs use rolling captions where each cue contains
    the previous line plus new text. This detects and removes overlaps.
    """
    if not segments:
        return segments

    deduped = [segments[0]]
    for seg in segments[1:]:
        prev_text = deduped[-1]["text"]
        curr_text = seg["text"]
        # Check if current text starts with previous text (rolling caption)
        if curr_text.startswith(prev_text):
            new_text = curr_text[len(prev_text):].strip()
            if new_text:
                deduped.append({"text": new_text, "start": seg["start"], "end": seg["end"]})
        elif prev_text in curr_text:
            # Previous text is somewhere in current — take only the new part after it
            idx = curr_text.index(prev_text) + len(prev_text)
            new_text = curr_text[idx:].strip()
            if new_text:
                deduped.append({"text": new_text, "start": seg["start"], "end": seg["end"]})
        else:
            deduped.append(seg)

    return deduped


def to_plaintext(segments, with_timestamps=False):
    """Convert segments to clean plaintext."""
    lines = []
    for seg in segments:
        if with_timestamps:
            ts = _format_timestamp(seg["start"])
            lines.append(f"[{ts}] {seg['text']}")
        else:
            lines.append(seg["text"])

    if with_timestamps:
        return "\n".join(lines)
    else:
        # Join into flowing text, collapsing whitespace
        text = " ".join(lines)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def _format_timestamp(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Extract YouTube video transcripts")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--lang", "-l", default="en", help="Language code (default: en)")
    parser.add_argument("--list-subs", action="store_true", help="List available subtitle languages")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON with timestamps")
    parser.add_argument("--with-timestamps", action="store_true", help="Include timestamps in plaintext")
    parser.add_argument("-o", "--output", help="Save to file instead of stdout")
    args = parser.parse_args()

    try:
        if args.list_subs:
            subs = list_subtitles(args.url)
            print(f"Title: {subs['title']}\n")
            if subs["manual"]:
                print("Manual subtitles:")
                for lang, exts in sorted(subs["manual"].items()):
                    print(f"  {lang}: {', '.join(exts)}")
            else:
                print("Manual subtitles: none")
            print()
            if subs["automatic"]:
                print(f"Auto-generated captions: {len(subs['automatic'])} languages")
                for lang in sorted(subs["automatic"].keys()):
                    print(f"  {lang}")
            else:
                print("Auto-generated captions: none")
            return

        segments, metadata = extract_transcript(args.url, args.lang)

        if args.json_output:
            output = json.dumps(
                {"metadata": metadata, "segments": segments},
                indent=2,
                ensure_ascii=False,
            )
        else:
            output = to_plaintext(segments, with_timestamps=args.with_timestamps)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Saved to {args.output}", file=sys.stderr)
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
