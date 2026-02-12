import re
from pathlib import Path
from config import DATA_DIR

LOG_FILE = Path(DATA_DIR) / "logs" / "wiki_full_log_1.rtf"  # change if needed
BATCH_SIZE = 25_000  # rows per batch

def count_successful_uploads(log_path: str):
    # RTF encodes checkmark ✓ as \uc0\u8713, but we'll match flexible patterns
    # New format: [Upload] ✓ +X,XXX chunks / Y,YYY rows (cumulative: Z,ZZZ chunks, W,WWW rows)
    # Old format: [Upload] ✓ +X,XXX (cumulative: Y,YYY) or just [Upload] X docs → ES
    
    pattern_new = re.compile(r"\[Upload\].*?cumulative:\s+([\d,]+)\s+chunks,\s+([\d,]+)\s+rows")
    pattern_upload = re.compile(r"\[Upload\].*?docs")

    total_chunks = 0
    total_rows = 0
    batches = 0
    format_detected = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Try new format first (with row counts)
            match = pattern_new.search(line)
            if match:
                total_chunks = int(match.group(1).replace(",", ""))
                total_rows = int(match.group(2).replace(",", ""))
                batches += 1
                format_detected = "new"
                continue
            
            # Fall back to any [Upload] line (old format or current run without cumulative yet)
            if pattern_upload.search(line):
                batches += 1
                if format_detected is None:
                    format_detected = "old"

    # For old format, calculate exact rows from batch count
    if format_detected == "old" and batches > 0:
        total_rows = batches * BATCH_SIZE

    return total_chunks, total_rows, batches, format_detected


if __name__ == "__main__":
    path = Path(LOG_FILE)

    if not path.exists():
        print(f"File not found: {path}")
        exit(1)

    chunks, rows, batches, fmt = count_successful_uploads(path)

    print("==== Upload Summary ====")
    print(f"Successful batches  : {batches}")
    print(f"Total chunks uploaded: {chunks:,}" if chunks > 0 else "Total chunks uploaded: (not tracked in this log format)")
    
    if fmt == "new":
        print(f"Total rows processed : {rows:,}")
        print(f"\n→ Use skip_rows={rows} to resume from this point")
    elif fmt == "old" and batches > 0:
        print(f"Total rows processed : {rows:,} (= {batches} batches × {BATCH_SIZE:,})")
        print(f"\n→ Use skip_rows={rows} to resume from this point")
    else:
        print("\n⚠ No upload records found in log.")
        print("The log might be from an incomplete run or doesn't contain [Upload] lines yet.")