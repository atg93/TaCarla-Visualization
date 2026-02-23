#!/usr/bin/env bash
set -euo pipefail

# ---------- USER CONFIG ----------
export MAIN_PATH="/media/hdd/text_data/deneme_data"
export TOWN_FOLDER="Town12"
export OUT_DIR="pyspark_vis_out"
export PYSPARK_ENABLE=True

IMG="pyspark.sif"
APP_DIR="${APP_DIR:-/home/tg22/remote-pycharm/Vis_TaCarla}"
APP_FILE="${APP_FILE:-_UI.py}"

PORT="${PORT:-8505}"


# Default to loopback (best for SSH tunnel)
ADDR="${ADDR:-127.0.0.1}"

# ---------- USER CONFIG ----------





# ---------- SANITY CHECKS (HOST) ----------
[ -f "$APP_DIR/$APP_FILE" ] || { echo "Error: cannot find $APP_FILE in $APP_DIR"; exit 1; }
[ -f "$IMG" ] || { echo "Error: container image not found at $IMG"; exit 1; }

# ---------- FREE THE PORT (HOST) ----------
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is in use. Killing existing process..."
  pkill -f "streamlit run .* --server.port $PORT" || true
  sleep 1
fi

# ---------- BIND MOUNTS ----------
BIND_ARGS=( --bind /datasets,/workspace,/media )
[ -d "/media/ssd/workspace" ] && BIND_ARGS+=( --bind /media/ssd/workspace:/media/ssd/workspace )

# ---------- STREAMLIT BROWSER HINTS (HOST) ----------
export STREAMLIT_BROWSER_SERVER_ADDRESS="localhost"
export STREAMLIT_SERVER_ADDRESS="${ADDR}"

# ---------- PASS VARS INTO CONTAINER ----------
# Any host var exported as SINGULARITYENV_<NAME> becomes <NAME> inside the container.
export SINGULARITYENV_MAIN_PATH="$MAIN_PATH"
export SINGULARITYENV_TOWN_FOLDER="$TOWN_FOLDER"
export SINGULARITYENV_OUT_DIR="$OUT_DIR"
export SINGULARITYENV_PYSPARK_ENABLE="$PYSPARK_ENABLE"
# (Optional) also pass Streamlit vars into the container if desired:
export SINGULARITYENV_STREAMLIT_BROWSER_SERVER_ADDRESS="$STREAMLIT_BROWSER_SERVER_ADDRESS"
export SINGULARITYENV_STREAMLIT_SERVER_ADDRESS="$STREAMLIT_SERVER_ADDRESS"

# ---------- FRIENDLY URL ----------
echo "-------------------------------------------"
echo "Please open  http://localhost:${PORT}  on the same machine"
echo "-------------------------------------------"

# ---------- RUN APP IN CONTAINER ----------
exec singularity exec --cleanenv --nv \
  "${BIND_ARGS[@]}" \
  "$IMG" \
  bash -lc "
    set -euo pipefail
    cd \"$APP_DIR\"
    ls -l \"$APP_FILE\" >/dev/null
    # Optional: ensure streamlit exists in the container
    command -v streamlit >/dev/null 2>&1 || { echo 'Streamlit not found in container'; exit 1; }
    exec streamlit run \"$APP_FILE\" \
      --server.headless true \
      --server.address \"$ADDR\" \
      --server.port \"$PORT\" \
      --browser.gatherUsageStats false
  "
