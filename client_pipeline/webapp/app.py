import asyncio
import json
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response
from landmark_pipeline import LandmarkPosturePipeline

PORT = 8600
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


def serve_static(request):
    path = request.path.split("?")[0]
    if path == "/":
        path = "/index.html"
    file_path = (FRONTEND_DIR / path.lstrip("/")).resolve()
    if FRONTEND_DIR not in file_path.parents and file_path != FRONTEND_DIR:
        return Response(403, "Forbidden", Headers(), b"forbidden")
    if not file_path.is_file():
        return Response(404, "Not Found", Headers(), b"not found")

    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    body = file_path.read_bytes()
    headers = Headers()
    headers["Content-Type"] = content_type
    headers["Content-Length"] = str(len(body))
    headers["Cross-Origin-Opener-Policy"] = "same-origin"
    headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    return Response(200, "OK", headers, body)


def process_request(connection, request):
    if request.headers.get("Upgrade", "").lower() == "websocket":
        return None
    return serve_static(request)


async def ws_handler(websocket):
    pipeline = None
    print("client connected")
    try:
        async for raw in websocket:
            msg = json.loads(raw)
            if msg["type"] == "init":
                pipeline = LandmarkPosturePipeline(
                    msg["exercise"],
                    countdown_sec=float(msg.get("countdown_sec", 5.0)),
                    class_confidence_threshold=float(msg.get("confidence_threshold", 0.0)),
                )
                await websocket.send(json.dumps({"type": "ready"}))
                continue

            if msg["type"] == "landmarks" and pipeline is not None:
                try:
                    result = pipeline.process_landmarks(msg["landmarks"], msg["t"], msg.get("fps", 30.0))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    continue
                await websocket.send(json.dumps({"type": "result", **result}))
    except websockets.exceptions.ConnectionClosed:
        pass
    print("client disconnected")


async def main():
    async with websockets.serve(ws_handler, "0.0.0.0", PORT, process_request=process_request, max_size=2 ** 20):
        print(f"Jalan di http://localhost:{PORT} (halaman web + backend, 1 port yg sama)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
