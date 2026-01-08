import asyncio
import logging
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from video_tracks import RTSPVideoStreamTrack # Import the shared RTSP track

logger = logging.getLogger(__name__)

# Global set to keep track of active PeerConnections to prevent garbage collection
active_peer_connections = set()

async def webrtc_stream_handler(request: web.Request) -> web.Response:
    """
    Handles incoming WebRTC offers, sets up a PeerConnection, adds an RTSP video track,
    and returns the WebRTC answer.
    """
    try:
        data = await request.json()
        offer = RTCSessionDescription(sdp=data["sdp"], type=data["type"])
        rtsp_url = data.get("source")

        if not rtsp_url:
            logger.error("No 'source' (RTSP URL) provided in WebRTC offer.")
            return web.Response(status=400, text="Missing 'source' parameter.")

        logger.info(f"Received WebRTC offer for RTSP stream: {rtsp_url}")

        pc = RTCPeerConnection()
        active_peer_connections.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            """Logs connection state changes and cleans up PeerConnection."""
            logger.info(f"PeerConnection state for {rtsp_url}: {pc.connectionState}")
            if pc.connectionState in ["failed", "closed", "disconnected"]:
                logger.info(f"Closing PeerConnection for {rtsp_url} due to state: {pc.connectionState}")
                await pc.close()
                active_peer_connections.discard(pc)

        try:
            print("HEHE")
            video_track = RTSPVideoStreamTrack(rtsp_url, config_path="./configs/base.yaml")
            pc.addTrack(video_track)
        except ConnectionError as e:
            logger.error(f"Could not initialize RTSP stream for {rtsp_url}: {e}")
            await pc.close() # Close PC if track cannot be added
            active_peer_connections.discard(pc)
            return web.Response(status=500, text=f"Failed to open RTSP stream: {e}")

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info(f"Sending WebRTC answer for {rtsp_url}")
        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type
        })
    except asyncio.CancelledError:
        logger.info("WebRTC stream handler cancelled.")
        raise
    except Exception as e:
        logger.exception("Error in WebRTC stream handler:")
        return web.Response(status=500, text=f"Internal server error: {e}")