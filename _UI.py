# _UI.py — Top-first layout, reliable Play, zero-loss preview
# Hides image immediately after switching episode; Stop halts instantly (no extra frame)
import os
from pathlib import Path
import re
import base64
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np  # noqa: F401
import time

pyspark_enable = os.getenv("PYSPARK_ENABLE", "false").strip().lower() in {"1","true","yes","on","y"}

if pyspark_enable:
    from run_pyspark import run_pyspark
else:
    from run_multiprocess import process_folder


#from run_multiprocess import process_folder
#pyspark_enable = False
# ================= BOOT =================
st.set_page_config(page_title="TaCarla Label Viewer", layout="wide")
st.title("TaCarla Label Viewer")
# (Tip caption removed)

# ===== Defaults =====
MAIN_PATH = os.getenv("MAIN_PATH")  #"/media/hdd/text_data"
TOWN_FOLDER = os.getenv("TOWN_FOLDER")
SENSORS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_sensors"
LABELS_PATH = MAIN_PATH + "/" + TOWN_FOLDER + "_labels"
DEFAULT_OUT_DIR   = MAIN_PATH + "/" + os.getenv("OUT_DIR")

# --------- CONFIG ---------
EXPECTED_FOLDERS = [
    "back", "back_left", "back_right",
    "front", "front_left", "front_right",
    "bev", "label", "lidar",
]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

# --------- SESSION DEFAULTS ---------
ss = st.session_state
ss.setdefault("DATA_PATH", SENSORS_PATH)
ss.setdefault("OUT_DIR", DEFAULT_OUT_DIR)
ss.setdefault("episode_idx", 0)
ss.setdefault("playing", False)
ss.setdefault("play_fps", 5)
ss.setdefault("_was_playing", False)  # for instant stop edge detection
# Always keep viewer focus (no UI toggle)
ss["keep_viewer_focus"] = True

# ------- Top-of-page alert area (shows RUNNING/DONE/ERROR states) -------
alert_box = st.empty()
# If a previous run left a state, reflect it
if ss.get("job_status") == "running":
    started = ss.get("job_started_at", time.time())
    alert_box.warning(f"PySpark is running… elapsed {time.time()-started:.1f}s")
elif ss.get("job_status") == "done":
    alert_box.success(ss.get("job_message", "PySpark finished."))
elif ss.get("job_status") == "error":
    alert_box.error(ss.get("job_message", "PySpark failed."))

# --------- HELPERS ---------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def _is_image(p: Path) -> bool:
    try:
        return p.is_file() and p.suffix.lower() in IMAGE_EXTS
    except Exception:
        return False

@st.cache_data(show_spinner=False)
def index_images_strict(root_dir: str):
    root = Path(root_dir)
    by_folder = {f: [] for f in EXPECTED_FOLDERS}
    total = 0
    for folder in EXPECTED_FOLDERS:
        sub = root / folder
        if not sub.exists():
            continue
        files = [p for p in sub.iterdir() if _is_image(p)]
        files.sort(key=lambda p: natural_key(p.name))
        by_folder[folder] = files
        total += len(files)
    return by_folder, total

@st.cache_data(show_spinner=False)
def index_images_recursive(root_dir: str, max_files: int = 5000):
    root = Path(root_dir)
    hits = []
    try:
        for p in root.rglob("*"):
            if _is_image(p):
                hits.append(p)
                if len(hits) >= max_files:
                    break
    except Exception:
        pass
    by_folder = {}
    for p in hits:
        folder = p.parent.name
        by_folder.setdefault(folder, []).append(p)
    for k in by_folder:
        by_folder[k].sort(key=lambda p: natural_key(p.name))
    return by_folder, len(hits)

def _request_scroll():
    ss["_scroll_to_viewer"] = True

def _do_scroll_if_requested(anchor_id: str = "viewer-anchor"):
    if ss.get("keep_viewer_focus") or ss.get("_scroll_to_viewer"):
        script = """
        <script>
        (function(){
          function scrollNow(){
            const doc = parent.document;
            const el = doc.getElementById("__ANCHOR__");
            if (!el) return;
            const y = el.getBoundingClientRect().top + parent.window.pageYOffset - 4;
            parent.window.scrollTo({ top: y, left: 0, behavior: "instant" });
          }
          scrollNow();
          let tries = 0;
          const id = parent.window.setInterval(function(){
            tries += 1; scrollNow(); if (tries > 10) parent.window.clearInterval(id);
          }, 50);
          const mo = new parent.window.MutationObserver(scrollNow);
          mo.observe(parent.document.body, { subtree: true, childList: true });
          parent.window.setTimeout(()=>{ try{ mo.disconnect(); }catch(e){} }, 1200);
        })();
        </script>
        """.replace("__ANCHOR__", anchor_id)
        components.html(script, height=0, scrolling=False)
        ss["_scroll_to_viewer"] = False

def set_preview_by_index(images, idx, do_rerun=True):
    if not images:
        return
    n = len(images)
    idx = (int(idx) % n + n) % n
    ss["current_index"] = idx
    ss["preview_path"] = str(images[idx])
    _request_scroll()
    if do_rerun:
        st.rerun()

def handle_keyboard(current):
    script = """
    <div id="focus-root" style="position:fixed; inset:0; pointer-events:none;">
      <input id="keytrap" aria-hidden="true"
             style="opacity:0; position:absolute; left:-9999px; top:-9999px;" />
    </div>
    <script>
      const trap = document.getElementById('keytrap');
      function ensureFocus(){ try { trap && trap.focus(); } catch(e){} }
      window.addEventListener('load', ensureFocus, {once:true});
      document.addEventListener('visibilitychange', ensureFocus);
      window.addEventListener('mousemove', ensureFocus, {passive:true});
      window.addEventListener('click', ensureFocus, {passive:true});
      setInterval(ensureFocus, 1200);

      const isEditable = el => {
        if (!el) return false;
        const t = el.tagName;
        if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || el.isContentEditable) return true;
        const role = el.getAttribute && el.getAttribute('role');
        if (role === 'slider' || role === 'spinbutton') return true;
        return false;
      };

      let lastSent = 0;
      function post(val){
        const now = Date.now();
        if (now - lastSent < 120) return;
        lastSent = now;
        window.parent.postMessage(
          { isStreamlitMessage: true, type: 'streamlit:setComponentValue', value: val }, '*'
        );
      }

      function onKeyDown(e){
        const active = document.activeElement;
        if (isEditable(active)) return;
        if (e.code === 'Space') { e.preventDefault(); if (!e.repeat) post('next'); }
        else if (e.key === 'ArrowRight') { if (!e.repeat) post('next'); }
        else if (e.key === 'ArrowLeft')  { if (!e.repeat) post('prev'); }
      }
      document.addEventListener('keydown', onKeyDown, {passive:false});
    </script>
    """
    action = components.html(script, height=0, scrolling=False)

    idx = ss.get("current_index", 0)
    if action == "next":
        set_preview_by_index(current, idx + 1, do_rerun=True)
    elif action == "prev":
        set_preview_by_index(current, idx - 1, do_rerun=True)

# ---------- ZERO-LOSS IMAGE RENDERING ----------
def _stat_sig(p: Path):
    try:
        stt = p.stat()
        return (stt.st_size, stt.st_mtime_ns)
    except Exception:
        return (0, 0)

@st.cache_data(show_spinner=False)
def _file_to_dataurl(path_str: str, stat_sig):
    """Return a data URL for the original file bytes (no re-encode).
       If TIFF, convert once to PNG to ensure browser display.
    """
    p = Path(path_str)
    ext = p.suffix.lower()
    if ext in (".tif", ".tiff"):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype == np.uint16:
            img8 = (img / 257).astype(np.uint8)
        else:
            img8 = img
        enc_ok, buf = cv2.imencode(".png", img8)
        if not enc_ok:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    else:
        try:
            raw = p.read_bytes()
        except Exception:
            return None
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(ext, "application/octet-stream")
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

def render_image_from_file(path_str: str, caption: str = "", sticky: bool = True):
    data_url = _file_to_dataurl(path_str, _stat_sig(Path(path_str)))
    if not data_url:
        st.warning("Could not load image.")
        return
    sticky_css = "position:sticky; top:8px;" if sticky else ""
    html = f"""
    <figure style="margin:0; {sticky_css}">
      <img src="{data_url}" style="display:block; width:100%; height:auto; image-rendering:auto;" />
      <figcaption style="font:12px/1.4 system-ui, sans-serif; color:#6b7280; padding-top:6px;">
        {caption}
      </figcaption>
    </figure>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------- MISC ----------
def list_episodes(parent: Path):
    eps = [p for p in parent.iterdir() if p.is_dir()]
    eps.sort(key=lambda p: natural_key(p.name))
    return eps

def dir_tree(path: Path, depth: int = 2, max_entries: int = 200):
    out = []
    def walk(p: Path, d: int, budget: list[int]):
        if d < 0 or budget[0] <= 0: return
        try:
            entries = sorted(p.iterdir(), key=lambda x: x.name)
        except Exception:
            return
        for e in entries:
            if budget[0] <= 0: break
            rel = e.relative_to(path)
            prefix = "  " * (2 - d)
            if e.is_dir():
                out.append(f"{prefix}📁 {rel}/"); budget[0] -= 1; walk(e, d - 1, budget)
            else:
                out.append(f"{prefix}🖼 {rel}" if _is_image(e) else f"{prefix}📄 {rel}"); budget[0] -= 1
    walk(path, depth, [max_entries])
    return "\n".join(out)

# ================== CORE STATE ==================
DATA_PATH = ss["DATA_PATH"]
OUT_DIR   = ss["OUT_DIR"]
src_root  = Path(DATA_PATH)
out_root  = Path(OUT_DIR)
out_root.mkdir(parents=True, exist_ok=True)

if not src_root.exists():
    st.error(f"Source root does not exist: {src_root}")
    st.stop()

episodes = list_episodes(src_root)
if not episodes:
    st.warning(f"No episode folders found under: {src_root}")
    st.stop()

# Track episode change to clear preview immediately
current_idx = ss.get("episode_idx", 0)
if "_last_episode_idx" not in ss:
    ss["_last_episode_idx"] = current_idx

# ================== TOP: CONTROL BAR + IMAGE ==================
st.markdown("<div id='viewer-anchor'></div>", unsafe_allow_html=True)
st.subheader("Pick episode and control playback")

top_left, top_right = st.columns([1.05, 1.35], gap="large")

with top_left:
    # Episode selector
    st.selectbox(
        "Episode",
        list(range(len(episodes))),
        index=current_idx,
        format_func=lambda i: episodes[i].name,
        key="episode_idx"
    )

    # If the user switched episodes, clear preview & stop play (hide image)
    if ss["episode_idx"] != ss["_last_episode_idx"]:
        ss["_last_episode_idx"] = ss["episode_idx"]
        for k in ("folder_choice", "current_index", "preview_path", "play_next_at"):
            ss.pop(k, None)
        ss["playing"] = False

# Recompute with (possibly) updated episode_idx
idx = max(0, min(ss["episode_idx"], len(episodes) - 1))
current_episode = episodes[idx].name
current_root = out_root / current_episode

# Index (strict first, fallback)
by_folder, total = index_images_strict(str(current_root)) if current_root.exists() else ({}, 0)
strict_total = total
if total == 0 and current_root.exists():
    by_folder, total = index_images_recursive(str(current_root))
mode = "STRICT" if strict_total > 0 else ("FALLBACK" if current_root.exists() else "N/A")

# Folder ordering + current list
all_folder_names = list(by_folder.keys())
ordered_folders = [f for f in EXPECTED_FOLDERS if f in by_folder] + [f for f in all_folder_names if f not in EXPECTED_FOLDERS]
ss.setdefault("folder_choice", ordered_folders[0] if ordered_folders else None)
folder_choice = ss["folder_choice"]
current = by_folder.get(folder_choice, []) if folder_choice else []

# Init selection (only if we still have a preview_path cleared by episode switch)
ss.setdefault("current_index", 0)
if "preview_path" not in ss and current:
    # Do NOT auto-set preview on episode change; keep hidden until user acts
    pass

# Header info and controls
with top_left:
    st.markdown(f"**Scenario:** `{current_episode}`  •  **Images:** {total}  •  **Index mode:** {mode}")

    # Keyboard handler (Space/→/←)
    handle_keyboard(current)

    # Play / FPS controls
    play_cols = st.columns([1, 1.2])
    with play_cols[0]:
        st.toggle("Play", value=ss["playing"], key="playing")
    with play_cols[1]:
        ss["play_fps"] = st.slider("FPS", 1, 30, ss["play_fps"])

    # Detect edges to stop immediately
    just_stopped = ss.get("_was_playing", False) and (not ss.get("playing", False))
    just_started = (not ss.get("_was_playing", False)) and ss.get("playing", False)

    # Slider + filename
    if current:
        idx_img = ss.get("current_index", 0)
        slider_val = st.slider(
            "Sample",
            1, len(current),
            (idx_img + 1) if "preview_path" in ss else 1,
            key=f"sample_slider_{folder_choice}_{len(current)}"
        )
        # If user moved the slider, set preview
        if ("preview_path" not in ss) or ((slider_val - 1) != idx_img):
            set_preview_by_index(current, slider_val - 1)
        try:
            fname = Path(current[slider_val - 1]).name
        except Exception:
            fname = ""
        st.markdown(
            f"<div style='font-size:0.9rem;'>Image {slider_val} / {len(current)} — {fname}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No images in this folder (or no output yet).")

with top_right:
    # Sticky image at top-right — ZERO-LOSS rendering from file
    pp = ss.get("preview_path")
    if pp:
        try:
            try:
                rel = Path(pp).relative_to(current_root)
            except Exception:
                rel = Path(pp).name
            render_image_from_file(pp, caption=str(rel), sticky=True)
        except Exception as e:
            st.warning(f"Could not open preview: {e}")
    else:
        # After episode switch, there is intentionally no image until user picks one
        st.info("Select a folder and a sample to view the image.")

# --- Server-side autoplay (robust; only if preview exists)
if just_stopped:
    ss.pop("play_next_at", None)  # cancel any pending tick immediately
elif ss.get("playing", False) and current and ("preview_path" in ss):
    fps = max(1, int(ss.get("play_fps", 5)))
    now = time.time()
    next_at = ss.get("play_next_at", 0.0)
    if now >= next_at:
        ss["play_next_at"] = now + (1.0 / fps)
        idx_img = ss.get("current_index", 0)
        set_preview_by_index(current, idx_img + 1, do_rerun=True)
else:
    ss.pop("play_next_at", None)

# Remember last play state for edge detection
ss["_was_playing"] = ss.get("playing", False)

# ============== FOLDERS SECTION (UNDER BAR + IMAGE) ==============
st.divider()
st.subheader("Folders")
if ordered_folders:
    grid_cols = st.columns(min(len(ordered_folders), 8))
    for i, f in enumerate(ordered_folders):
        with grid_cols[i % len(grid_cols)]:
            if st.button(f"{f} ({len(by_folder.get(f, []))})", key=f"folders_btn_{f}"):
                ss["folder_choice"] = f
                for k in ("current_index", "preview_path"):
                    ss.pop(k, None)
                _request_scroll()
                st.rerun()
else:
    st.write("— No folders indexed yet —")

# ================== DETAILS (ALL OTHER INFO) BELOW ==================
st.divider()
st.header("Details & Actions")

# IO roots (editable)
io_cols = st.columns(2)
with io_cols[0]:
    new_data = st.text_input("Source root (episodes to process):", value=ss["DATA_PATH"]).strip()
    if new_data != ss["DATA_PATH"]:
        ss["DATA_PATH"] = new_data
        st.rerun()
with io_cols[1]:
    new_out = st.text_input("Output root (pipeline writes here):", value=ss["OUT_DIR"]).strip()
    if new_out != ss["OUT_DIR"]:
        ss["OUT_DIR"] = new_out
        st.rerun()

# Run PySpark (shows warning while running)
run_cols = st.columns([1, 2, 2])
with run_cols[0]:
    run_now = st.button("Visualize the current episode", type="primary")
with run_cols[1]:
    st.caption(f"Source: `{(Path(ss['DATA_PATH'])/current_episode).as_posix()}`")
with run_cols[2]:
    st.caption(f"Expecting output under: `{(Path(ss['OUT_DIR'])/current_episode).as_posix()}`")

if run_now:
    # Mark running & show warning immediately at the top
    ss["job_status"] = "running"
    ss["job_started_at"] = time.time()
    alert_box.warning("Placing labels on the image…")
    t0 = time.time()
    try:
        if pyspark_enable:
            with st.spinner("Running PySpark…"):
                run_pyspark(folder=current_episode, SENSORS_PATH=SENSORS_PATH, LABELS_PATH=LABELS_PATH, OUT_DIR=str(Path(ss['OUT_DIR'])))
        else:
            with st.spinner("Running Multiprocess…"):
                process_folder(folder=current_episode, SENSORS_PATH=SENSORS_PATH, LABELS_PATH=LABELS_PATH, OUT_DIR=str(Path(ss['OUT_DIR'])))
        dt = time.time() - t0
        ss["job_status"] = "done"
        if pyspark_enable:
            ss["job_message"] = f"PySpark finished (fallback) in {dt:.1f}s"
        else:
            ss["job_message"] = f"Multiprocess finished (fallback) in {dt:.1f}s"
        alert_box.success(ss["job_message"])
    except TypeError:
        if pyspark_enable:
            with st.spinner("Running PySpark (fallback signature)…"):
                run_pyspark(folder=current_episode, SENSORS_PATH=SENSORS_PATH, LABELS_PATH=LABELS_PATH)
        else:
            with st.spinner("Running Multiprocess (fallback signature)…"):
                process_folder(folder=current_episode, SENSORS_PATH=SENSORS_PATH, LABELS_PATH=LABELS_PATH, OUT_DIR=str(Path(ss['OUT_DIR'])))

        dt = time.time() - t0
        ss["job_status"] = "done"
        if pyspark_enable:
            ss["job_message"] = f"PySpark finished (fallback) in {dt:.1f}s"
        else:
            ss["job_message"] = f"Multiprocess finished (fallback) in {dt:.1f}s"
        alert_box.success(ss["job_message"])
    except Exception as e:
        ss["job_status"] = "error"
        ss["job_message"] = f"Pipeline failed: {e}"
        alert_box.error(ss["job_message"])
    st.cache_data.clear()
    st.rerun()

# Debug
with st.expander("Debug: paths, counts, directory tree", expanded=False):
    src_root = Path(ss["DATA_PATH"])
    out_root = Path(ss["OUT_DIR"])
    st.write({
        "src_root": str(src_root),
        "out_root": str(out_root),
        "current_root": str(current_root),
        "current_root_exists": current_root.exists(),
        "strict_total": strict_total,
        "total_indexed": total,
        "folders_indexed": {k: len(v) for k, v in by_folder.items()},
        "episode_idx": ss["episode_idx"],
        "last_episode_idx": ss.get("_last_episode_idx"),
        "has_preview": "preview_path" in ss,
        "playing": ss.get("playing", False),
        "job_status": ss.get("job_status"),
        "job_message": ss.get("job_message"),
    })
    st.text("Directory tree (depth=2):")
    if current_root.exists():
        st.code(dir_tree(current_root, depth=2, max_entries=300), language="text")
    else:
        st.code("(no output directory yet)", language="text")

# Auto-scroll back to top preview if needed
_do_scroll_if_requested()
