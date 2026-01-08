
Terminal 1: 
docker run --rm -it \
  -e MTX_RTSPADDRESS=:8555 \
  -e MTX_RTSPTRANSPORTS=tcp \
  -e MTX_WEBRTCADDITIONALHOSTS=192.168.1.106 \
  -p 8555:8555 \
  -p 7855:1935 \
  -p 7879:8888 \
  -p 7843:8889 \
  -p 7827:8890/udp \
  -p 7171:8189/udp \
  bluenviron/mediamtx

Terminal 2:     
ffmpeg -re -stream_loop -1 \
    -i data/input/fall_test/fall.mp4 \
    -c copy -f rtsp \
    -rtsp_transport tcp rtsp://127.0.0.1:8555/live


Terminal 3: python backend/main.py


rtsp://127.0.0.1:8555/live
http://127.0.0.1:2222