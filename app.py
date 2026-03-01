"""
Chef's Hat RL Agent - Web Interface

Flask-based web UI to run demonstrations and training from the browser.
Run: python app.py
Then open http://localhost:5000 in your browser.

Endpoints:
  GET  /           - Serves the single-page UI (HTML + CSS + JS)
  POST /api/demo   - Runs a demo (JSON body: { matches, model_path? })
  POST /api/train  - Runs training (JSON body: { matches })
"""

import os
import sys
import io
import json
import threading

# Suppress gym/deprecation warnings before any env imports
os.environ["GYM_SILENCE_WARNINGS"] = "1"
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path for local imports (demo, train_gym)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template_string, request, jsonify

# Flask app: single app, no blueprints
app = Flask(__name__)
# Limit request body size (e.g. for future file uploads)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024

# Shared state updated by background threads; read by API handlers
_run_result = {"status": "idle", "output": [], "summary": {}}
_run_lock = threading.Lock()


def _run_demo(matches=5, model_path=None):
    """
    Run a demonstration game in a background thread.
    Updates global _run_result with status, per-match output lines, and summary stats.
    """
    global _run_result
    with _run_lock:
        _run_result = {"status": "running", "output": [], "summary": {}}

    try:
        # NumPy 2.x compatibility for gym (bool8 removed)
        import numpy as np
        if not hasattr(np, "bool8"):
            np.bool8 = np.bool_

        from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
        from ChefsHatGym.env import ChefsHatEnv

        # Use demo module's agents: quiet randoms + RL agent (silent=True for web)
        from demo import QuietRandomAgent, DemoRLAgent

        out_dir = os.path.join(os.path.dirname(__file__), "interface_output")
        os.makedirs(out_dir, exist_ok=True)

        # Room: MATCHES mode, stop after `matches` matches, no dataset/logs
        room = ChefsHatRoomLocal(
            room_name="WebDemo",
            game_type=ChefsHatEnv.GAMETYPE["MATCHES"],
            stop_criteria=matches,
            max_rounds=-1,
            save_dataset=False,
            verbose_console=False,
            verbose_log=False,
            game_verbose_console=False,
            game_verbose_log=False,
            log_directory=out_dir,
        )
        room.add_player(QuietRandomAgent("R1", log_directory=out_dir))
        room.add_player(QuietRandomAgent("R2", log_directory=out_dir))
        room.add_player(QuietRandomAgent("R3", log_directory=out_dir))
        rl = DemoRLAgent("RL", log_directory=out_dir, model_path=model_path, silent=True)
        room.add_player(rl)

        room.start_new_game()

        # Build response: one line per match, plus aggregate summary
        win_rate = rl.win_count / max(1, rl.match_count) * 100
        avg_pos = float(np.mean(rl.positions)) if rl.positions else 0
        output = [f"Match {i+1}: {['1st','2nd','3rd','4th'][p-1]} place" for i, p in enumerate(rl.positions)]
        summary = {"win_rate": win_rate, "avg_position": avg_pos, "matches": rl.match_count}

        with _run_lock:
            _run_result = {"status": "done", "output": output, "summary": summary}
    except Exception as e:
        with _run_lock:
            _run_result = {"status": "error", "output": [str(e)], "summary": {}}


def _run_training(matches=100):
    """
    Run training in a background thread.
    Delegates to train_gym.run_training; updates _run_result with output and summary.
    """
    global _run_result
    with _run_lock:
        _run_result = {"status": "running", "output": [], "summary": {}}

    try:
        from train_gym import run_training
        out_dir = os.path.join(os.path.dirname(__file__), "interface_output", "train")
        room, agent = run_training(matches=int(matches), seed=42, output_dir=out_dir)
        win_rate = agent.win_count / max(1, agent.match_count) * 100
        avg_pos = float(__import__("numpy").mean(agent.positions)) if agent.positions else 0
        output = [f"Training complete: {matches} matches"]
        summary = {"win_rate": win_rate, "avg_position": avg_pos, "matches": agent.match_count}
        with _run_lock:
            _run_result = {"status": "done", "output": output, "summary": summary}
    except Exception as e:
        import traceback
        with _run_lock:
            _run_result = {"status": "error", "output": [str(e), traceback.format_exc()], "summary": {}}


# -----------------------------------------------------------------------------
# Single-page UI: embedded HTML template (no separate template files)
# -----------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chef's Hat RL Agent</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      min-height: 100vh;
      color: #e8e8e8;
      padding: 2rem;
    }
    .container {
      max-width: 700px;
      margin: 0 auto;
    }
    h1 {
      font-size: 1.75rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
      background: linear-gradient(90deg, #e94560, #f39c12);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .subtitle {
      color: #888;
      font-size: 0.9rem;
      margin-bottom: 2rem;
    }
    .card {
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1.5rem;
    }
    .card h2 {
      font-size: 1rem;
      font-weight: 600;
      color: #a0a0a0;
      margin-bottom: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
    }
    input[type="number"] {
      width: 80px;
      padding: 0.5rem 0.75rem;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,0.2);
      background: rgba(0,0,0,0.3);
      color: #fff;
      font-size: 0.95rem;
    }
    input[type="number"]:focus {
      outline: none;
      border-color: #e94560;
    }
    button {
      padding: 0.6rem 1.2rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: transform 0.1s, opacity 0.2s;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .btn-demo {
      background: linear-gradient(135deg, #e94560, #c73e54);
      color: white;
    }
    .btn-train {
      background: linear-gradient(135deg, #0f3460, #16213e);
      color: white;
      border: 1px solid rgba(255,255,255,0.2);
    }
    .output-box {
      background: rgba(0,0,0,0.4);
      border-radius: 8px;
      padding: 1rem;
      min-height: 120px;
      max-height: 300px;
      overflow-y: auto;
      font-family: 'Consolas', 'Monaco', monospace;
      font-size: 0.85rem;
      line-height: 1.6;
      color: #b8d4e3;
    }
    .output-box .line { margin-bottom: 0.25rem; }
    .output-box .line.match-1 { color: #4ade80; }
    .output-box .line.match-2 { color: #94a3b8; }
    .output-box .line.match-3 { color: #fbbf24; }
    .output-box .line.match-4 { color: #f87171; }
    .output-box.empty { color: #666; }
    .summary {
      display: flex;
      gap: 1.5rem;
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid rgba(255,255,255,0.1);
    }
    .summary-item {
      font-size: 0.9rem;
    }
    .summary-item strong { color: #e94560; }
    .status {
      font-size: 0.85rem;
      color: #94a3b8;
      margin-top: 0.5rem;
    }
    .status.running { color: #fbbf24; }
    .status.error { color: #f87171; }
    .status.done { color: #4ade80; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Chef's Hat RL Agent</h1>
    <p class="subtitle">Reinforcement Learning — Multi-Agent Card Game</p>

    <div class="card">
      <h2>Demonstration</h2>
      <div class="controls">
        <label>
          Matches: <input type="number" id="demoMatches" value="5" min="1" max="50">
        </label>
        <button class="btn-demo" id="btnDemo">Run Demo</button>
      </div>
      <p class="status" id="demoStatus"></p>
    </div>

    <div class="card">
      <h2>Training</h2>
      <div class="controls">
        <label>
          Matches: <input type="number" id="trainMatches" value="100" min="10" max="1000">
        </label>
        <button class="btn-train" id="btnTrain">Start Training</button>
      </div>
      <p class="status" id="trainStatus"></p>
    </div>

    <div class="card">
      <h2>Output</h2>
      <div class="output-box" id="output">
        <span class="empty">Run a demo or training to see results.</span>
      </div>
      <div class="summary" id="summary" style="display:none;"></div>
    </div>
  </div>

  <script>
    // Demo and Train buttons POST to /api/demo and /api/train; display output lines and summary (win rate, avg position)
    const output = document.getElementById('output');
    const summary = document.getElementById('summary');

    function setOutput(html, isSummary = false) {
      output.innerHTML = html;
      output.classList.toggle('empty', !html);
      summary.style.display = isSummary ? 'flex' : 'none';
    }

    function setStatus(elId, text, cls) {
      const el = document.getElementById(elId);
      el.textContent = text;
      el.className = 'status ' + (cls || '');
    }

    async function runAction(action) {
      const btnDemo = document.getElementById('btnDemo');
      const btnTrain = document.getElementById('btnTrain');
      btnDemo.disabled = true;
      btnTrain.disabled = true;
      setOutput('<span class="empty">Running...</span>');
      setStatus('demoStatus', action === 'demo' ? 'Running demo...' : '');
      setStatus('trainStatus', action === 'train' ? 'Training...' : '');

      try {
        const url = action === 'demo' ? '/api/demo' : '/api/train';
        const matches = action === 'demo'
          ? document.getElementById('demoMatches').value
          : document.getElementById('trainMatches').value;
        const res = await fetch(url, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({matches: parseInt(matches)})
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);

        let html = '';
        if (data.output && data.output.length) {
          data.output.forEach((line, i) => {
            const m = line.match(/Match \\d+: (\\d)(st|nd|rd|th)/);
            const cls = m ? 'match-' + m[1] : '';
            html += '<div class="line ' + cls + '">' + escapeHtml(line) + '</div>';
          });
        }
        setOutput(html || '<span class="empty">No output.</span>');

        if (data.summary && Object.keys(data.summary).length) {
          const s = data.summary;
          summary.innerHTML = `
            <div class="summary-item"><strong>Win rate:</strong> ${(s.win_rate || 0).toFixed(1)}%</div>
            <div class="summary-item"><strong>Avg position:</strong> ${(s.avg_position || 0).toFixed(2)}</div>
            <div class="summary-item"><strong>Matches:</strong> ${s.matches || 0}</div>
          `;
          summary.style.display = 'flex';
        }
        setStatus('demoStatus', action === 'demo' ? 'Done' : '', 'done');
        setStatus('trainStatus', action === 'train' ? 'Done' : '', 'done');
      } catch (err) {
        setOutput('<span class="empty" style="color:#f87171">' + escapeHtml(err.message) + '</span>');
        setStatus('demoStatus', 'Error', 'error');
        setStatus('trainStatus', 'Error', 'error');
      }
      btnDemo.disabled = false;
      btnTrain.disabled = false;
    }

    function escapeHtml(s) {
      const div = document.createElement('div');
      div.textContent = s;
      return div.innerHTML;
    }

    document.getElementById('btnDemo').onclick = () => runAction('demo');
    document.getElementById('btnTrain').onclick = () => runAction('train');
  </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the single-page application (HTML with embedded CSS and JS)."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/demo", methods=["POST"])
def api_demo():
    """
    POST /api/demo: run a demonstration.
    Body: { "matches": int, "model_path": str (optional) }
    Runs _run_demo in a thread and returns JSON: { output, summary } or { error }.
    """
    try:
        data = request.get_json() or {}
        matches = int(data.get("matches", 5))
        model_path = data.get("model_path")
        thread = threading.Thread(target=_run_demo, args=(matches, model_path))
        thread.start()
        thread.join(timeout=300)  # 5 min max
        if thread.is_alive():
            return jsonify({"error": "Timeout (5 min)"})
        with _run_lock:
            r = _run_result.copy()
        if r["status"] == "error":
            return jsonify({"error": "\\n".join(r["output"])})
        return jsonify({"output": r["output"], "summary": r["summary"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    POST /api/train: run training.
    Body: { "matches": int }
    Runs _run_training in a thread; returns JSON with output and summary or error.
    """
    try:
        data = request.get_json() or {}
        matches = int(data.get("matches", 100))
        thread = threading.Thread(target=_run_training, args=(matches,))
        thread.start()
        thread.join(timeout=600)  # 10 min max
        if thread.is_alive():
            return jsonify({"error": "Timeout (10 min)"})
        with _run_lock:
            r = _run_result.copy()
        if r["status"] == "error":
            return jsonify({"error": "\\n".join(r["output"])})
        return jsonify({"output": r["output"], "summary": r["summary"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Bind to all interfaces so it's reachable from other devices on the network
    print("\n  Chef's Hat RL Agent - Web Interface")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
