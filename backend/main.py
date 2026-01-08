import os
import asyncio
import logging
from aiohttp import web

# Configure logging for the entire application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import handlers from separate modules
from webrtc_handler import webrtc_stream_handler, active_peer_connections
from websocket_handler import websocket_handler, clients # Import clients for potential shutdown logic
import yaml

def create_app(frontend_path: str = "frontend") -> web.Application:
    """
    Creates and configures the aiohttp web application.
    """
    app = web.Application()

    # Add WebRTC route
    app.router.add_post("/webrtc/stream", webrtc_stream_handler)
    logger.info("Registered WebRTC stream handler at /webrtc/stream")
    
    # Add WebSocket route
    # app.router.add_get("/ws/stream", websocket_handler)
    # logger.info("Registered WebSocket stream handler at /ws/stream")
    
    # Add control endpoint to stop active WebRTC streams
    async def stop_webrtc_streams(request: web.Request) -> web.Response:
        try:
            closed = 0
            for pc in list(active_peer_connections):
                if pc.connectionState not in ["closed"]:
                    await pc.close()
                active_peer_connections.discard(pc)
                closed += 1
            logger.info(f"Stopped {closed} active WebRTC peer connections.")
            return web.json_response({"stopped": closed})
        except Exception as e:
            logger.exception("Failed to stop WebRTC streams:")
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_post("/webrtc/stop", stop_webrtc_streams)
    logger.info("Registered stop handler at /webrtc/stop")

    # Add configuration save endpoint
    async def save_config(request: web.Request) -> web.Response:
        try:
            data = await request.json()
            yaml_text = data.get("yaml")
            if not yaml_text:
                return web.json_response({"error": "Missing 'yaml' in human"}, status=400)
            # Validate YAML structure
            try:
                parsed = yaml.safe_load(yaml_text)
                if not isinstance(parsed, dict):
                    return web.json_response({"error": "YAML must represent a mapping/object"}, status=400)
            except Exception as e:
                return web.json_response({"error": f"Invalid YAML: {e}"}, status=400)

            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            logger.info(f"Configuration saved to {config_path}")
            return web.json_response({"status": "ok"})
        except Exception as e:
            logger.exception("Failed to save configuration:")
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_post("/config", save_config)
    logger.info("Registered config save handler at /config")

    # Add configuration fetch endpoint
    async def get_config(request: web.Request) -> web.Response:
        try:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
            if not os.path.isfile(config_path):
                return web.json_response({"error": "config not found"}, status=404)
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Return raw text for YAML so frontend can edit easily
            return web.Response(text=content, content_type='text/plain')
        except Exception as e:
            logger.exception("Failed to read configuration:")
            return web.json_response({"error": str(e)}, status=500)

    app.router.add_get("/config", get_config)
    logger.info("Registered config get handler at /config")
    
    # Serve static files from the 'frontend' directory
    abs_frontend_path = os.path.abspath(frontend_path)
    if not os.path.isdir(abs_frontend_path):
        logger.warning(f"Frontend directory not found at {abs_frontend_path}. Static files might not be served.")
    else:
        app.router.add_static("/", path=abs_frontend_path, show_index=True)
        logger.info(f"Serving static files from: {abs_frontend_path}")

    return app

async def main():
    """Main function to run the aiohttp application."""
    host = "192.168.2.21"
    port = 2222
    frontend_dir = "frontend"

    app = create_app(frontend_path=frontend_dir)
    logger.info(f"Starting aiohttp server on http://{host}:{port}/index.html")
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    # Keep the server running indefinitely
    try:
        while True:
            await asyncio.sleep(3600) # Sleep for 1 hour, or until interrupted
    except asyncio.CancelledError:
        logger.info("Server shutdown initiated.")
    finally:
        logger.info("Performing cleanup before server stops...")
        # Clean up active PeerConnections
        for pc in list(active_peer_connections): # Iterate over a copy as set might be modified
            logger.info(f"Closing lingering PeerConnection: {pc.connectionState}")
            await pc.close()
        active_peer_connections.clear()

        # Clean up active WebSocket clients (though handlers should mostly handle this)
        for session_id, ws in list(clients.items()):
            logger.info(f"Closing lingering WebSocket for client: {session_id}")
            if not ws.closed:
                await ws.close()
        clients.clear()

        await runner.cleanup()
        logger.info("Server stopped.")


if __name__ == "__main__":
    # Ensure OpenCV environment variables are set before any cv2.VideoCapture calls in modules
    # This is handled within video_tracks.py, but good to ensure global setup is consistent if moved.
    os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)


