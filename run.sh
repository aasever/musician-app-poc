#!/usr/bin/env bash
# Start the Musician App server.
# Usage: ./run.sh [port]   (default port 8000)

PORT=${1:-8000}

echo "Installing / checking dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting Musician App on http://localhost:${PORT}"
echo "Open that URL in your browser to use the app."
echo ""

exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
