#!/usr/bin/env bash
# Serve the self-contained report locally for PRIVATE viewing — no public host, no pushes
# (matches world-commander-bench/scripts/serve_report.sh). report.html embeds all images +
# replay frames, so a plain static server is enough.
#
# On amax41:    bash experiments/serve_report.sh            # serves on 127.0.0.1:8899
# On your Mac:  ssh -L 8899:localhost:8899 <amax41-host>
#               then open http://localhost:8899/report.html
set -euo pipefail
cd "$(dirname "$0")"            # the experiments/ dir, where report.html lives
PORT="${1:-8899}"
echo "Serving $(pwd) at http://127.0.0.1:${PORT}/report.html"
echo "From your Mac: ssh -L ${PORT}:localhost:${PORT} <amax41>  then open http://localhost:${PORT}/report.html"
exec python3 -m http.server "$PORT" --bind 127.0.0.1
