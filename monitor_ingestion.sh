#!/bin/bash
# Monitor ingestion progress

while true; do
    clear
    echo "=== INGESTION MONITOR - $(date) ==="
    echo ""
    
    # Check if process is running
    if [ -f logs/ingestion.pid ]; then
        PID=$(cat logs/ingestion.pid)
        if ps -p $PID > /dev/null 2>&1; then
            echo "✅ Process running (PID: $PID)"
            ps -p $PID -o etime,pcpu,pmem | tail -1
        else
            echo "❌ Process not running"
        fi
    else
        echo "⚠️  No PID file found"
    fi
    
    echo ""
    echo "Status File:"
    if [ -f data/raw/.ingestion_status.json ]; then
        python3 << 'PYTHON'
import json
try:
    with open('data/raw/.ingestion_status.json') as f:
        d = json.load(f)
    print(f"  Current Step: {d.get('current_step', 'N/A')}")
    print(f"  Completed: {', '.join(d.get('completed_steps', []))}")
    p = d.get('progress', {}).get('minor_league_pitching', {})
    if p:
        print(f"  Progress: {p.get('current', 0)}/{p.get('total', '?')} ({p.get('percent', 0):.1f}%)")
except Exception as e:
    print(f"  Error reading status: {e}")
PYTHON
    else
        echo "  No status file yet"
    fi
    
    echo ""
    echo "Output Files:"
    ls -lh data/raw/*.{parquet,jsonl} 2>/dev/null | awk '{printf "  %-45s %8s\n", $9, $5}' || echo "  No files yet"
    
    echo ""
    echo "JSONL Progress:"
    if [ -f data/raw/minor_league_pitching.jsonl ]; then
        LINES=$(wc -l < data/raw/minor_league_pitching.jsonl 2>/dev/null)
        SIZE=$(du -h data/raw/minor_league_pitching.jsonl 2>/dev/null | cut -f1)
        echo "  Records: $LINES"
        echo "  Size: $SIZE"
    else
        echo "  File not created yet"
    fi
    
    echo ""
    echo "Recent Logs (last 3 lines):"
    tail -3 logs/ingestion.log 2>/dev/null | sed 's/^/  /' || echo "  No logs yet"
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
