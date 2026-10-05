"""JSON CLI and a local HTTP adapter for the exact same Planner."""
import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from .model import Ranker
from .planner import Planner, load_config


def request_plan(planner, value):
    if "observation" in value:
        return planner.plan(value["observation"], value.get("candidates"),
            value.get("mode", "ahead"), value.get("allow_buffer", False), value.get("execution_mode", "proposal"),
            value.get("buffer_candidates"))
    return planner.plan(value)


def make_server(planner, host="127.0.0.1", port=8080):
    lock = Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health": self.send_error(404); return
            self.reply(200, dict(status="ok", model_status=planner.model_status, version="ahead-action-2.1"))

        def do_POST(self):
            if self.path not in ("/plan", "/evaluate-sequence"): self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 2_000_000: raise ValueError("Request body must be 1..2000000 bytes")
                value = json.loads(self.rfile.read(length))
                with lock:
                    if self.path == "/evaluate-sequence":
                        from .sequence import evaluate_sequence
                        result = evaluate_sequence(value["observation"], value["steps"], planner.config)
                    else: result = request_plan(planner, value)
                self.reply(200, result)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                self.reply(400, dict(error=str(error), action=None))

        def reply(self, status, value):
            data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, fmt, *args): pass
    return ThreadingHTTPServer((host, port), Handler)


def serve(planner, host="127.0.0.1", port=8080):
    server = make_server(planner, host, port)
    print(f"AHEAD listening on http://{host}:{server.server_port}; model={planner.model_status}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "serve", "teach", "evaluate-sequence", "visualize"])
    parser.add_argument("--input"); parser.add_argument("--output")
    parser.add_argument("--config"); parser.add_argument("--model"); parser.add_argument("--weights")
    parser.add_argument("--base-group", default="external"); parser.add_argument("--split", default="train")
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else None
    if args.weights:
        if config is None: config = Planner.released().config
        config["weights"] = json.loads(Path(args.weights).read_text(encoding="utf-8"))
    model = Ranker.load(args.model) if args.model else None
    planner = Planner(config, model)
    if args.command == "serve": serve(planner, args.host, args.port); return
    value = json.loads(Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read())
    if args.command == "visualize":
        from .visualize import render_svg
        if not args.output: parser.error("visualize requires --output ending in .svg")
        result = value.get("result") or request_plan(planner, value)
        Path(args.output).write_text(render_svg(value.get("observation", value), result), encoding="utf-8")
        return
    if args.command == "evaluate-sequence":
        from .sequence import evaluate_sequence
        result = evaluate_sequence(value["observation"], value["steps"], planner.config)
    elif args.command == "teach":
        from .teacher import label_observation
        result = label_observation(value.get("observation", value), planner.config, value.get("candidates"), args.base_group, args.split)
    else: result = request_plan(planner, value)
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output: Path(args.output).write_text(text, encoding="utf-8")
    else: print(text)


if __name__ == "__main__": main()
