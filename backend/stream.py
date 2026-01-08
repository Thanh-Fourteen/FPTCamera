import os
import asyncio
import logging
import sys

from webrtc_handler import webrtc_stream_handler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from aiohttp import web

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set OpenCV environment variables for better RTSP handling
os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"



def create_app(frontend_path: str = "frontend") -> web.Application:
    """
    Creates and configures the aiohttp web application.
    """
    app = web.Application()
    app.router.add_post("/webrtc/stream", webrtc_stream_handler)
    
    # Serve static files from the 'frontend' directory
    # Ensure 'frontend' directory exists and contains index.html
    abs_frontend_path = os.path.abspath(frontend_path)
    if not os.path.isdir(abs_frontend_path):
        logger.warning(f"Frontend directory not found at {abs_frontend_path}. Static files might not be served.")
        # Optionally, create a dummy index.html or raise an error if frontend is mandatory
        # For now, just log a warning.
    else:
        app.router.add_static("/", path=abs_frontend_path, show_index=True)
        logger.info(f"Serving static files from: {abs_frontend_path}")

    return app

async def main():
    """Main function to run the aiohttp application."""
    host = "127.0.0.1"
    port = 6868
    frontend_dir = "frontend"

    app = create_app(frontend_path=frontend_dir)
    logger.info(f"Starting aiohttp server on http://{host}:{port}")
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
        await runner.cleanup()
        logger.info("Server stopped.")



if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user (Ctrl+C).")
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)