#!/usr/bin/env python3
"""
Codex -> OpenAI-compatible API Bridge for Koharu
Exposes an OpenAI-compatible /v1/chat/completions endpoint powered by your local
logged-in ChatGPT/Codex subscription via `codex app-server`.
"""

import json
import logging
import re
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import threading
import time

sys.path.insert(0, "/home/keyfer/bilingual_book_maker")
from book_maker.codex_client import CodexAppServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("codex-bridge")

HOST = "127.0.0.1"
PORT = 8765

class CodexManager:
    def __init__(self):
        self.server = None
        self.lock = threading.Lock()
        self._init_server()

    def _init_server(self):
        logger.info("Iniciando conexión con Codex app-server...")
        self.server = CodexAppServer()
        self.server.start()
        limits = self.server.ensure_logged_in()
        logger.info(f"Codex conectado exitosamente. Plan: {limits.plan_type}, Uso: {limits.used_percent}%")

    def run_completion(self, system_prompt, user_content, model=None):
        # Mapear modelos reconocidos por Codex app-server
        valid_models = {"gpt-6.1-sol", "gpt-6-sol", "gpt-6-astra", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra"}
        codex_model = model if model in valid_models else None
        with self.lock:
            try:
                thread_id = self.server.start_thread(
                    model=codex_model,
                    base_instructions=system_prompt if system_prompt else None
                )
                logger.info(f"Thread {thread_id} creado con modelo {codex_model or 'default (gpt-6.1-sol)'}. Enviando solicitud a Codex...")
                response_text = self.server.run_turn(thread_id, user_content, timeout=120)
                return response_text
            except Exception as e:
                logger.warning(f"Error en turn ({e}). Reiniciando servidor Codex...")
                try:
                    self.server.close()
                except Exception:
                    pass
                self._init_server()
                thread_id = self.server.start_thread(
                    model=codex_model,
                    base_instructions=system_prompt if system_prompt else None
                )
                return self.server.run_turn(thread_id, user_content, timeout=120)

codex_mgr = CodexManager()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

class OpenAIBridgedHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/v1", "/health"):
            self._send_json(200, {
                "status": "ok",
                "service": "Koharu Codex Bridge",
                "provider": "codex-app-server",
                "endpoints": {
                    "models": "/v1/models",
                    "chat": "/v1/chat/completions"
                }
            })
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        elif path in ("/models", "/v1/models"):
            models = [
                {"id": "gpt-6-luna", "object": "model", "owned_by": "openai"},
                {"id": "gpt-6.1-sol", "object": "model", "owned_by": "openai"},
                {"id": "gpt-6-astra", "object": "model", "owned_by": "openai"},
                {"id": "gpt-5.6-luna", "object": "model", "owned_by": "openai"}
            ]
            self._send_json(200, {"object": "list", "data": models})
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/chat/completions", "/v1/chat/completions"):
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            messages = payload.get("messages", [])
            model = payload.get("model")

            system_parts = []
            user_parts = []

            for m in messages:
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, list):
                    # Multimodal parts: extract text
                    texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                    content = "\n".join(texts)
                elif not isinstance(content, str):
                    content = str(content or "")

                if role == "system":
                    system_parts.append(content)
                elif role in ("user", "assistant"):
                    user_parts.append(content)

            system_prompt = "\n\n".join(system_parts)
            user_content = "\n\n".join(user_parts)

            try:
                answer = codex_mgr.run_completion(system_prompt, user_content, model=model)
                
                # Limpiar bloques markdown si vienen rodeados de ```json
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", answer)
                clean_answer = match.group(1).strip() if match else answer.strip()

                response = {
                    "id": f"chatcmpl-codex-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model or "gpt-5-codex",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": clean_answer
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(user_content) // 4,
                        "completion_tokens": len(clean_answer) // 4,
                        "total_tokens": (len(user_content) + len(clean_answer)) // 4
                    }
                }
                self._send_json(200, response)
            except Exception as e:
                logger.error(f"Error procesando completion: {e}", exc_info=True)
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Not Found"})

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} - {format % args}")

def run():
    server = ThreadedHTTPServer((HOST, PORT), OpenAIBridgedHandler)
    logger.info(f"Servidor Codex OpenAI Bridge escuchando en http://{HOST}:{PORT}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        try:
            codex_mgr.server.close()
        except Exception:
            pass

if __name__ == "__main__":
    run()
