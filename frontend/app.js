class SmartCityApp {
    constructor() {
        this.defaultConfig = {
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S",
                "max_bytes": 10485760,
                "backup_count": 5,
                "debug_mode": false
            },
            "input": {
                "rotate_degrees": 90,
                "source": "data/input/goryokaku_tower-cam-006_20250503_090006_4-6_300s_rotate.mp4",
                "camera_id": "4-6",
                "area_name": "Goryokaku Tower 1F_Observation ELV hall"
            },
            "pipeline": {
                "target_fps": -1,
                "skip_frames": -1,
                "codec": "h264",
                "profile_times": false,
                "buffer_strategy": "fixed",
                "buffer_size": 30,
                "visualize": "detection",
                "output_path": "data/output/goryokaku_tower-cam-006_20250503_090006_4-6_300s_rotate.mp4",
                "timestamp": "2025-05-03 09:00:06",
                "modules": ["human_detector", "tracker", "human_counter"]
            },
            "modules": {
                "human_detector": {
                    "onnx_path": "weights/best.onnx",
                    "threshold": 0.2,
                    "iou_threshold": 0.45,
                    "img_size": 640,
                    "class_names": {
                        "0": "human",
                    }
                },
                "tracker": {
                    "strategy": "bytetrack",
                    "track_thresh": 0.3,
                    "match_thresh": 0.8,
                    "track_buffer": 60,
                    "use_iou_score_only": false,
                },
                "human_counter": {
                    "strategy": "humancount",
                    "visualize": true,
                    "include_classes": ["human"]
                },
            },
            "database": {
                "filepath": "logs/smartcity_events.db"
            },
            "application": {
                "default_camera_id": "unknown_camera",
                "default_area_name": "unknown_area",
                "exit_codes": {
                    "config_error": 1,
                    "media_source_error": 2,
                    "pipeline_error": 3
                },
                "frame_producer": {
                    "reconnect_attempts": 3,
                    "reconnect_timeout_s": 5
                }
            }
        };

        this.selectOptions = {
            "logging.default_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "application.environment": ["production", "staging", "development", "test"],
            "application.visualize": ["detection", "tracking"],
            "application.input.buffer_strategy": ["fixed", "priority", "dynamic"],
            "application.input.buffer_drop_policy": ["drop_oldest", "drop_newest"],
            "application.input.codec": ["h264", "h265", "vp9"],
            "modules.tracker.strategy": ["bytetrack", "sort", "deepsort"],
        };

        this.configMode = 'simple';
        this.advancedKeys = new Set();

        this.config = JSON.parse(JSON.stringify(this.defaultConfig));
        // Chỉ cho phép các module được chỉ định
        this.allowedModules = ['human_detector', 'tracker', 'human_counter'];
        this.allModules = this.allowedModules;
        this.updatePipelineModulesUI = null;

        this.stage = null;
        this.layer = null;
        this.imageLayer = null;
        this.backgroundImage = null;
        this.currentPolygon = null;
        this.currentArea = null;
        this.polygons = [];
        this.isDrawing = false;
        this.zoom = 1;
        this.actionHistory = [];
        this.actionIndex = -1;
        this.currentJobId = null;
        this.pollingInterval = null;
        this.videoFile = null;
        this.streamUrl = null;
        this.backendUrl= null;
        this.currentVideoProcessingMode = 'upload';
        this.renderedVideoUrl = null;

        this.selectedGroup = null;
        this.drawingAreaType = null;
        this.hiddenAreaIdentifiers = new Set();
        this.isErasing = false;
        this.snapThreshold = 15;
        this.snapIndicator = null;

        // --- Debounced function for saving the session ---
        this.saveSessionDebounced = this.debounce(this.saveSession, 400);

        // Token to cancel in-flight stage creation (avoids reopening canvas after close)
        this.stageCreationSeq = 0;

        this.init();
    }

    // --- Debounce helper to prevent rapid function calls ---
    debounce(func, delay) {
        let timeout;
        return function (...args) {
            const context = this;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), delay);
        };
    }

    // --- Helper function to format YAML with proper quoting ---
    formatYamlWithQuotes(config) {
        const yamlString = jsyaml.dump(config, { 
            quotingType: '"',
            forceQuotes: false,
            styles: {
                '!!str': 'literal'
            },
            lineWidth: -1, // Prevent line wrapping
            noRefs: true   // Prevent references
        });
        
        return yamlString;
    }

    // --- Normalize current UI config to match standard configs/base.yaml schema ---
    normalizeToBaseConfig(rawConfig) {
        const cfg = JSON.parse(JSON.stringify(rawConfig || {}));

        // Ensure objects exist
        cfg.logging = cfg.logging || {};
        cfg.input = cfg.input || {};
        cfg.pipeline = cfg.pipeline || {};
        cfg.modules = cfg.modules || {};
        cfg.application = cfg.application || {};

        // application
        const application = {
            app_name: cfg.application.app_name || 'FPTCameraApp',
            environment: cfg.application.environment || 'development',
            camera_id: cfg.application.camera_id || cfg.input.camera_id || 'camera-01',
            area_name: cfg.application.area_name || cfg.input.area_name || 'outdoor',
            visualize: cfg.application.visualize || cfg.pipeline.visualize || 'tracking',
            input: {
                source: cfg.input.source ?? 'data/input/fall_test/fall.mp4',
                rotate_degrees: cfg.input.rotate_degrees ?? 0,
                target_fps: cfg.pipeline.target_fps ?? -1,
                skip_frames: cfg.pipeline.skip_frames ?? -1,
                buffer_size: cfg.pipeline.buffer_size ?? 1,
                buffer_strategy: cfg.pipeline.buffer_strategy || 'fixed',
                buffer_drop_policy: cfg.application?.input?.buffer_drop_policy || cfg.input.buffer_drop_policy || 'drop_oldest',
                codec: cfg.pipeline.codec || cfg.application?.input?.codec || 'h264',
            },
            output: {
                output_path: (cfg.application?.output?.output_path) || cfg.pipeline.output_path || 'data/output/test.mp4',
            },
            frame_producer: {
                reconnect_attempts: cfg.application?.frame_producer?.reconnect_attempts ?? 3,
                reconnect_timeout_s: cfg.application?.frame_producer?.reconnect_timeout_s ?? 5,
            },
        };

        // logging
        const logging = {
            default_level: (cfg.logging.default_level || cfg.logging.level || 'INFO').toString().toUpperCase(),
            main_log_file: cfg.logging.main_log_file || 'logs/app.log',
            log_file_max_bytes: cfg.logging.log_file_max_bytes ?? cfg.logging.max_bytes ?? 10485760,
            log_file_backup_count: cfg.logging.log_file_backup_count ?? cfg.logging.backup_count ?? 5,
            debug_mode: cfg.logging.debug_mode !== undefined ? !!cfg.logging.debug_mode : true,
        };
        // Preserve optional formatting keys if present
        if (cfg.logging.format) logging.format = cfg.logging.format;
        if (cfg.logging.date_format) logging.date_format = cfg.logging.date_format;
        if (cfg.logging.specialized_loggers) logging.specialized_loggers = cfg.logging.specialized_loggers;

        // pipeline flags: map enabled modules to booleans
        const pipeline = {};
        const enabledList = Array.isArray(cfg.pipeline.modules) ? cfg.pipeline.modules : [];
        const allKnownModules = this.allowedModules || ['human_detector', 'tracker', 'human_counter'];
        allKnownModules.forEach(name => {
            // Prefer explicit boolean if present; otherwise, fallback to modules list
            if (typeof cfg.pipeline[name] === 'boolean') {
                pipeline[name] = !!cfg.pipeline[name];
            } else {
                pipeline[name] = enabledList.includes(name);
            }
        });

        // modules: carry over as-is for known modules only
        const modules = {};
        allKnownModules.forEach(name => {
            if (cfg.modules[name]) modules[name] = cfg.modules[name];
        });

        // Ensure human_counter.coordinates is array
        if (!modules.human_counter) modules.human_counter = { strategy: 'humancount', visualize: true, include_classes: ['human'] };
        if (!Array.isArray(modules.human_counter.coordinates)) modules.human_counter.coordinates = [];

        // Build final base config object
        const base = {
            application,
            logging,
            pipeline,
            modules,
        };

        // Pass-through optional sections supported by some environments
        if (cfg.database) base.database = cfg.database;

        return base;
    }

    // --- Sync polygon coordinates across System Configuration, Module Configuration, and Human Counter
    _syncCoordinates(coords) {
        const safe = Array.isArray(coords) ? coords : [];
        if (!this.config.modules) this.config.modules = {};
        if (!this.config.modules.human_counter) this.config.modules.human_counter = {};
        this.config.modules.human_counter.coordinates = safe;
        // Mirror to top-level for System Configuration view
        this.config.coordinates = safe;
        // Mirror to Modules root so coordinates can be surfaced if needed
        this.config.modules.coordinates = safe;

        // Live-update any visible inputs bound to coordinates while Settings is open
        const pathsToSync = [
            'modules.human_counter.coordinates',
            'coordinates',
            'modules.coordinates'
        ];

        const containers = [
            'configForm', 'modulesConfigForm', 'pipelineConfigForm',
            'loggingConfigForm', 'databaseConfigForm', 'advancedConfigForm'
        ];

        const valueString = JSON.stringify(safe);
        containers.forEach(containerId => {
            const container = document.getElementById(containerId);
            if (!container) return;
            pathsToSync.forEach(path => {
                const input = container.querySelector(`.field-input[data-path="${path}"]`);
                if (input) {
                    if (input.type === 'checkbox') {
                        // Not expected for coordinates, but keep defensive
                        input.checked = Array.isArray(safe) && safe.length > 0;
                    } else if (input.type === 'range' || input.type === 'number') {
                        // Not expected; skip numeric
                    } else {
                        input.value = valueString;
                    }
                }
            });
        });
    }

    // --- Helper function to merge backend config with defaults ---
    mergeConfigWithDefaults(backendConfig) {
        const merged = JSON.parse(JSON.stringify(this.defaultConfig));
        
        // Deep merge backend config into defaults, but only for allowed modules
        const deepMerge = (target, source) => {
            for (const key in source) {
                if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                    if (key === 'modules') {
                        // Chỉ merge các module được phép
                        if (!target[key]) target[key] = {};
                        for (const moduleKey in source[key]) {
                            if (this.allowedModules.includes(moduleKey)) {
                                if (!target[key][moduleKey]) target[key][moduleKey] = {};
                                deepMerge(target[key][moduleKey], source[key][moduleKey]);
                            }
                        }
                    } else {
                        if (!target[key]) target[key] = {};
                        deepMerge(target[key], source[key]);
                    }
                } else {
                    target[key] = source[key];
                }
            }
        };
        
        deepMerge(merged, backendConfig);
        
        // Ensure critical modules have required properties
        if (!merged.modules.human_counter) {
            merged.modules.human_counter = { strategy: 'humancount', visualize: true, include_classes: ['human'], coordinates: [] };
        } else {
            if (!Array.isArray(merged.modules.human_counter.coordinates)) {
                merged.modules.human_counter.coordinates = [];
            }
        }
        
        // Sync pipeline.modules array from boolean flags (source of truth)
        if (!merged.pipeline) merged.pipeline = {};
        const allowed = this.allowedModules || ['human_detector', 'tracker', 'human_counter'];
        merged.pipeline.modules = allowed.filter(name => merged.pipeline && merged.pipeline[name] === true);
        
        return merged;
    }

    init() {
        this.loadSession();
        this.generateConfigForm();
        this.setupEventListeners();
        this.log('System initialized', 'success');
        this.updateZoomDisplay();
        this.updateToolState();
        this.updateUndoRedoButtons();
        this.updateModeToggleUI();
        this.handleVideoProcessingModeChange(); // Gọi khi khởi tạo để thiết lập trạng thái ban đầu
        this.initializeSessionStatus(); // Khởi tạo trạng thái session status icon
    }

    loadCurrentCamera() {
        try {
            const currentCamera = localStorage.getItem('currentCamera');
            if (currentCamera) {
                const camera = JSON.parse(currentCamera);
                this.log(`Loaded camera: ${camera.name}`, 'info');
                
                // Update stream URL input if available
                const streamUrlInput = document.getElementById('streamUrlInput');
                if (streamUrlInput && camera.url) {
                    streamUrlInput.value = camera.url;
                }
                
                // Update camera ID in config
                if (camera.id) {
                    this.config.input.camera_id = camera.id;
                }
                
                // Update area name if provided
                if (camera.location) {
                    this.config.input.area_name = camera.location;
                }
                
                // Clear the current camera from localStorage after loading
                localStorage.removeItem('currentCamera');
            }
        } catch (error) {
            console.error('Error loading current camera:', error);
            this.log('Error loading camera data', 'error');
        }
    }

    setupEventListeners() {
        const streamButton = document.getElementById('streamButton');
        const streamUrlInput = document.getElementById('streamUrlInput');
        const stopStreamButton = document.getElementById('stopStreamButton');
        const openSettingsBtn = document.getElementById('openSettings');
        const closeSettingsBtn = document.getElementById('closeSettings');
        const cancelSettingsBtn = document.getElementById('cancelSettings');
        const saveSettingsBtn = document.getElementById('saveSettings');
        const settingsModal = document.getElementById('settingsModal');
        const settingsBackdrop = document.getElementById('settingsBackdrop');
        const modeYamlBtn = document.getElementById('modeYaml');
        const modeFormBtn = document.getElementById('modeForm');
        const yamlEditor = document.getElementById('yamlEditor');
        const drawAreaBtn = document.getElementById('drawAreaBtn');
        const clearAreaBtn = document.getElementById('clearAreaBtn');
        const addAreaBtn = document.getElementById('addArea');
        const closeCanvasBtn = document.getElementById('closeCanvas');
    
        if (streamButton) {
            streamButton.addEventListener('click', () => this.startStream());
        }
        if (stopStreamButton) {
            stopStreamButton.addEventListener('click', () => this.stopStream());
        }
    
        if (streamUrlInput && streamButton) {
            streamUrlInput.addEventListener('input', () => {
                streamButton.disabled = !this.isValidUrl(streamUrlInput.value);
            });
        }
    
        // Nếu có backendUrlInput
        const backendUrlInput = document.getElementById('backendUrlInput');
        if (backendUrlInput) {
            backendUrlInput.addEventListener('input', () => {
                this.saveSessionDebounced?.();  
            });
        }

        // Add Area button event listener
        if (addAreaBtn) {
            addAreaBtn.addEventListener('click', async () => {
                // Close settings modal if it's open
                const settingsModal = document.getElementById('settingsModal');
                if (settingsModal && !settingsModal.classList.contains('hidden')) {
                    settingsModal.classList.add('hidden');
                }
                // Start drawing new area
                await this.startNewArea('human_counter');
            });
        }

        // Close Canvas button event listener
        if (closeCanvasBtn) {
            closeCanvasBtn.addEventListener('click', () => {
                this.closeDrawingCanvas();
                this.log('Drawing canvas closed by user.', 'info');
            });
        }

        // Back to home button event listener
        const backToHomeBtn = document.getElementById('backToHomeBtn');
        if (backToHomeBtn) {
            backToHomeBtn.addEventListener('click', () => {
                window.location.href = 'home.html';
            });
        }

        // Load camera from localStorage if available
        this.loadCurrentCamera();

        // Removed global draw/clear buttons. Controls now reside in Settings > Module Config > Human Counter.

        // Settings modal behaviors
        const openModal = async () => {
            if (!settingsModal) return;
            try {
                const backendUrl = document.getElementById('backendUrlInput')?.value?.trim();
                if (this.isValidUrl(backendUrl)) {
                    const res = await fetch(`${backendUrl}/config`);
                    if (res.ok) {
                        const yamlText = await res.text();
                        const parsed = jsyaml.load(yamlText);
                        if (parsed && typeof parsed === 'object') {
                            this.config = this.mergeConfigWithDefaults(parsed);
                        }
                        if (yamlEditor) yamlEditor.value = yamlText;
                    }
                }
            } catch (e) {
                this.log(`Failed to load config from backend: ${e.message}`, 'error');
            }
            settingsModal.classList.remove('hidden');
            this.generateConfigForm();
            this.initializeSettingsNavigation();
        };
        const closeModal = () => settingsModal?.classList.add('hidden');
        openSettingsBtn?.addEventListener('click', openModal);
        closeSettingsBtn?.addEventListener('click', closeModal);
        cancelSettingsBtn?.addEventListener('click', closeModal);
        settingsBackdrop?.addEventListener('click', closeModal);

        modeYamlBtn?.addEventListener('click', () => {
            document.getElementById('configForm')?.classList.add('hidden');
            yamlEditor?.classList.remove('hidden');
        });
        modeFormBtn?.addEventListener('click', () => {
            yamlEditor?.classList.add('hidden');
            document.getElementById('configForm')?.classList.remove('hidden');
        });

        saveSettingsBtn?.addEventListener('click', async () => {
            try {
                // If YAML editor is visible, parse it back
                if (!yamlEditor?.classList.contains('hidden')) {
                    const parsed = jsyaml.load(yamlEditor.value);
                    if (typeof parsed !== 'object') throw new Error('YAML must be an object');
                    this.config = parsed;
                } else {
                    // Otherwise collect from form controls
                    this.updateConfigFromForm();
                }
                
                // Always save to session first
                this.saveSessionDebounced();
                
                // Try to persist to backend if URL is valid
                const backendUrl = document.getElementById('backendUrlInput')?.value?.trim();
                if (this.isValidUrl(backendUrl)) {
                    try {
                        const normalized = this.normalizeToBaseConfig(this.config);
                        const res = await fetch(`${backendUrl}/config`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ yaml: this.formatYamlWithQuotes(normalized) })
                        });
                        if (!res.ok) throw new Error(await res.text());
                        this.log('Configuration saved to backend.', 'success');
                    } catch (backendError) {
                        this.log(`Backend save failed: ${backendError.message}. Configuration saved locally.`, 'warning');
                    }
                } else {
                    this.log('Configuration saved locally (no backend URL provided).', 'success');
                }
                
                closeModal();
            } catch (e) {
                this.log(`Failed to save configuration: ${e.message}`, 'error');
            }
        });
    }
    
    handleVideoProcessingModeChange() {
        const modeUploadFile = document.getElementById('modeUploadFile');
        const modeStream = document.getElementById('modeStream');
        const videoUploadContainer = document.getElementById('videoUploadContainer');
        const renderButton = document.getElementById('renderButton');
        const liveStreamArea = document.getElementById('videoStreamArea');
        const streamMethodContainer = document.getElementById('streamMethodContainer');

        if (modeUploadFile.checked) {
            this.currentVideoProcessingMode = 'upload';
            videoUploadContainer.classList.remove('hidden');
            liveStreamArea.classList.add('hidden'); // hiden stream when upload
            renderButton.disabled = !this.videoFile; // Active render if file is selected
            streamMethodContainer.classList.add('hidden');
            this.log('Switched to file upload mode.');
        } else if (modeStream.checked) {
            this.currentVideoProcessingMode = 'stream';
            videoUploadContainer.classList.add('hidden');
            streamMethodContainer.classList.remove('hidden');
            // renderButton.disabled = true;
            this.log('Switched to stream mode.');
        }
        
        document.getElementById('renderResult')?.classList.add('hidden');
        document.getElementById('renderProgress')?.classList.add('hidden');
    }

    // --- Hàm xử lý khi người dùng nhấn "Start Stream" ---
    startStream() {
        const streamUrlInput = document.getElementById('streamUrlInput');
        const backendUrlInput = document.getElementById('backendUrlInput');
        const streamStatus = document.getElementById('streamStatus');

        const streamUrl = streamUrlInput.value.trim();
        const backendUrl = backendUrlInput.value.trim();
        if (!this.isValidUrl(streamUrl)) {
            this.log('Please enter a valid stream URL.', 'error');
            return;
        }

        if (!this.isValidUrl(backendUrl)) {
            this.log('Please enter a valid backend URL.', 'error');
            return;
        }

        this.streamUrl = streamUrl;
        this.backendUrl = backendUrl;
        this.videoFile = null;
        this.log('Connecting to stream...', 'info');
        this.startWebRTCStream(streamUrl, backendUrl).then(() => {
            this.log('Stream connected.', 'success');
        }).catch(err => {
            this.log('Failed to load stream. Check URL.', 'error');
            this.log('WebRTC error: ' + err.message, 'error');
        });
    }

    async stopStream() {
        try {
            const backendUrlInput = document.getElementById('backendUrlInput');
            const backendUrl = backendUrlInput?.value?.trim();
            if (!this.isValidUrl(backendUrl)) {
                this.log('Please enter a valid backend URL.', 'error');
                return;
            }
            const res = await fetch(`${backendUrl}/webrtc/stop`, { method: 'POST' });
            if (!res.ok) throw new Error(await res.text());
            // Also stop video element tracks if any
            const videoElement = document.getElementById('liveStream');
            if (videoElement?.srcObject) {
                videoElement.srcObject.getTracks().forEach(t => t.stop());
                videoElement.srcObject = null;
            }
            this.log('Stopped stream.', 'success');
        } catch (e) {
            this.log(`Failed to stop stream: ${e.message}`, 'error');
        }
    }

    isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    async startWebRTCStream(rtspUrl, backendUrl) { // Ensure backendUrl is received here
        const videoElement = document.getElementById('liveStream');
        const pc = new RTCPeerConnection();

        pc.ontrack = (event) => {
            videoElement.srcObject = event.streams[0];
        };

        const canvas = Object.assign(document.createElement("canvas"), { width: 1, height: 1 });
        const stream = canvas.captureStream();
        stream.getTracks().forEach(track => pc.addTrack(track, stream));

        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        this.log(`Sending WebRTC offer to: ${backendUrl}/webrtc/stream`); 

        try {
            const res = await fetch(`${backendUrl}/webrtc/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type,
                    source: rtspUrl
                })
            });

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`HTTP error! Status: ${res.status}, Body: ${errorText}`);
            }

            const data = await res.json();
            await pc.setRemoteDescription(new RTCSessionDescription(data));
            this.log("WebRTC stream started successfully.", 'success');
        } catch (error) {
            this.log(`WebRTC stream failed: ${error.message}`, 'error');
            throw error;
        }
    }

    updateModeToggleUI() {
        document.querySelectorAll('.mode-toggle-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === this.configMode);
        });
    }

    getImageToCanvasCoordinates(x, y) {
        if (!this.backgroundImage || !this.backgroundImage.image()) return { x, y };

        const image = this.backgroundImage.image();
        const scaleX = this.backgroundImage.width() / image.naturalWidth;
        const scaleY = this.backgroundImage.height() / image.naturalHeight;

        const canvasX = (x * scaleX) + this.backgroundImage.x();
        const canvasY = (y * scaleY) + this.backgroundImage.y();

        return { x: canvasX, y: canvasY };
    }

    getCanvasToImageCoordinates(canvasX, canvasY) {
        if (!this.backgroundImage || !this.backgroundImage.image()) return { x: canvasX, y: canvasY };

        const image = this.backgroundImage.image();
        const scaleX = this.backgroundImage.width() / image.naturalWidth;
        const scaleY = this.backgroundImage.height() / image.naturalHeight;

        const relativeX = canvasX - this.backgroundImage.x();
        const relativeY = canvasY - this.backgroundImage.y();

        const imageX = relativeX / scaleX;
        const imageY = relativeY / scaleY;

        return { x: imageX, y: imageY };
    }

    getOrCreateKonvaOverlay() {
        // Ensure a Konva overlay exists on top of the live video
        let overlay = document.getElementById('konvaOverlay');
        if (!overlay) {
            const videoArea = document.getElementById('videoStreamArea');
            if (!videoArea) return null;
            overlay = document.createElement('div');
            overlay.id = 'konvaOverlay';
            videoArea.appendChild(overlay);
        }
        return overlay;
    }

    updateOverlayMappingFromVideo(videoEl, overlay) {
        const overlayRect = overlay.getBoundingClientRect();
        const videoRect = videoEl.getBoundingClientRect();

        // Lock intrinsic video size once available to avoid re-scaling of polygons
        // when metadata becomes available after initial rendering.
        if (!this._intrinsicVideoSize) {
            const intrinsicW = (videoEl.videoWidth && videoEl.videoWidth > 0) ? videoEl.videoWidth : (this.backgroundImage?.image?.().naturalWidth || displayW || 1280);
            const intrinsicH = (videoEl.videoHeight && videoEl.videoHeight > 0) ? videoEl.videoHeight : (this.backgroundImage?.image?.().naturalHeight || displayH || 720);
            this._intrinsicVideoSize = { width: intrinsicW, height: intrinsicH };
        } else {
            // If we didn't have real metadata before but now we do, only upgrade once
            if ((videoEl.videoWidth && videoEl.videoWidth > 0) && (videoEl.videoHeight && videoEl.videoHeight > 0)) {
                if (!this._intrinsicVideoSize._lockedWithReal) {
                    this._intrinsicVideoSize = { width: videoEl.videoWidth, height: videoEl.videoHeight, _lockedWithReal: true };
                }
            }
        }

        const vw = this._intrinsicVideoSize.width;
        const vh = this._intrinsicVideoSize.height;

        // Compute the actual displayed content box inside the video element
        // to account for letterboxing caused by object-fit: contain.
        const scaleToFit = Math.min(videoRect.width / vw, videoRect.height / vh);
        const contentW = vw * scaleToFit;
        const contentH = vh * scaleToFit;
        const contentLeft = videoRect.left + (videoRect.width - contentW) / 2;
        const contentTop = videoRect.top + (videoRect.height - contentH) / 2;

        const offsetX = (contentLeft - overlayRect.left);
        const offsetY = (contentTop - overlayRect.top);
        const displayW = contentW;
        const displayH = contentH;

        this.backgroundImage = {
            x: () => offsetX,
            y: () => offsetY,
            width: () => displayW,
            height: () => displayH,
            image: () => ({ naturalWidth: vw, naturalHeight: vh })
        };
    }

    async ensureKonvaStageFromVideo(videoEl) {
        // Show tools but keep stream visible (draw over it)
        const canvasContainerPre = document.getElementById('canvasContainer');
        if (canvasContainerPre && canvasContainerPre.classList.contains('hidden')) {
            canvasContainerPre.classList.remove('hidden');
        }

        const overlay = this.getOrCreateKonvaOverlay();
        if (!overlay) { this.log('Konva overlay not found.', 'error'); return; }

        // Size the stage to the overlay and compute the video box within overlay
        const overlayRect = overlay.getBoundingClientRect();
        const videoRect = videoEl.getBoundingClientRect();
        const widthVisible = overlayRect.width || 960;
        const heightVisible = overlayRect.height || 540;

        if (!this.stage) {
            this.stage = new Konva.Stage({ container: 'konvaOverlay', width: widthVisible, height: heightVisible, draggable: false });
            this.layer = new Konva.Layer();
            this.stage.add(this.layer);
            this.snapIndicator = new Konva.Circle({ x: 0, y: 0, radius: 8, fill: 'rgba(50,184,198,0.5)', visible: false });
            this.layer.add(this.snapIndicator);
        } else {
            this.stage.size({ width: widthVisible, height: heightVisible });
        }

        // No background image drawing; just maintain a mapping object for coordinate conversions
        if (this.imageLayer) this.imageLayer.destroy();
        this.imageLayer = new Konva.Layer();
        this.stage.add(this.imageLayer);
        this.imageLayer.moveToBottom();
        this.updateOverlayMappingFromVideo(videoEl, overlay);

        // Attach click handler for drawing on live video frame
        try { this.stage.off('click tap'); } catch (_) { /* no-op */ }
        this.stage.on('click tap', async () => {
            if (!this.isDrawing) return;
            const pos = this.stage.getPointerPosition();
            if (!pos) return;
            // Only accept clicks inside displayed video rect
            const offsetX = this.backgroundImage.x();
            const offsetY = this.backgroundImage.y();
            const displayW = this.backgroundImage.width();
            const displayH = this.backgroundImage.height();
            if (pos.x < offsetX || pos.y < offsetY || pos.x > offsetX + displayW || pos.y > offsetY + displayH) return;
            const imageCoords = this.getCanvasToImageCoordinates(pos.x, pos.y);
            await this.addPoint(imageCoords.x, imageCoords.y);
        });

        // Keep video visible and show tools
        document.getElementById('canvasContainer')?.classList.remove('hidden');
        this.drawExistingPolygons();

        // Keep mapping/stage in sync with layout changes
        const resizeHandler = () => {
            const rect = overlay.getBoundingClientRect();
            this.stage.size({ width: rect.width, height: rect.height });
            this.updateOverlayMappingFromVideo(videoEl, overlay);
            this.drawExistingPolygons();
        };
        try {
            if (!this._overlayResizeBound) {
                this._overlayResizeBound = true;
                window.addEventListener('resize', resizeHandler);
                const ro = new ResizeObserver(resizeHandler);
                ro.observe(overlay);
                ro.observe(videoEl);
                this._overlayRO = ro;
            }
        } catch (_) { /* optional */ }
    }

    drawExistingPolygons() {
        if (!this.layer) return;
        this.layer.destroyChildren();
        if (this.snapIndicator) this.layer.add(this.snapIndicator);
        this.polygons = [];
        this.selectedGroup = null;


        // Chỉ vẽ các module được phép
        if (this.config.modules?.human_counter?.visualize && this.config.modules.human_counter.coordinates?.length > 2) {
            this.createPolygonFromCoordinates(this.config.modules.human_counter.coordinates, 'Human Counting Area', -1, 'human_counter');
        }

        this.updateScaleDependentElements();
        this.updateToolState();
    }

    createPolygonFromCoordinates(coordinates, name, areaIndex, areaType) {
        if (!this.layer) return;

        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot create polygon for module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        const identifier = `${areaType}_${areaIndex}`;
        const isVisible = !this.hiddenAreaIdentifiers.has(identifier);

        // Detect if coordinates are normalized (0..1) and convert to image pixels for drawing
        const isNormalized = Array.isArray(coordinates)
            && coordinates.length > 0
            && coordinates.every(c => Array.isArray(c) && c.length === 2 && c[0] >= 0 && c[0] <= 1 && c[1] >= 0 && c[1] <= 1);

        let imageCoords = coordinates;
        if (isNormalized && this.backgroundImage && this.backgroundImage.image) {
            const img = this.backgroundImage.image();
            const imgW = img.naturalWidth;
            const imgH = img.naturalHeight;
            imageCoords = coordinates.map(c => [c[0] * imgW, c[1] * imgH]);
        }

        const canvasCoords = imageCoords.map(coord => this.getImageToCanvasCoordinates(coord[0], coord[1]));
        const points = canvasCoords.flatMap(p => [p.x, p.y]);

        const color = this.getColorForIndex(areaIndex);

        const polygon = new Konva.Line({
            points: points,
            fill: `rgba(${color}, 0.3)`,
            stroke: `rgb(${color})`,
            strokeWidth: 2,
            closed: true,
            draggable: false
        });

        const labelPos = canvasCoords[0];
        const label = new Konva.Text({
            x: labelPos.x,
            y: labelPos.y - 20,
            text: name,
            fontSize: 14,
            fontFamily: 'Arial',
            fill: `rgb(${color})`,
            backgroundColor: 'rgba(255,255,255,0.7)',
            padding: 4,
            listening: false,
        });

        const group = new Konva.Group({ areaIndex, areaType, visible: isVisible });
        group.add(polygon, label);

        canvasCoords.forEach((p, vertexIndex) => {
            const vertexHandle = new Konva.Circle({
                x: p.x,
                y: p.y,
                radius: 6,
                fill: 'blue',
                stroke: 'white',
                strokeWidth: 1,
                draggable: false,
                name: 'vertex-handle',
                visible: false,
            });

            vertexHandle.on('dragmove', () => this.updatePolygonOnVertexDrag(group, vertexIndex));
            vertexHandle.on('dragend', () => this.updateConfigOnVertexDragEnd(group));

            vertexHandle.on('mouseenter', () => {
                if (vertexHandle.isDragging() || this.isErasing) return;
                this.stage.container().style.cursor = 'move';
                vertexHandle.to({ scaleX: 1.5, scaleY: 1.5, duration: 0.1 });
            });

            vertexHandle.on('mouseleave', () => {
                if (vertexHandle.isDragging() || this.isErasing) return;
                this.stage.container().style.cursor = 'pointer';
                vertexHandle.to({ scaleX: 1, scaleY: 1, duration: 0.1 });
            });

            vertexHandle.on('click tap', async (e) => {
                e.cancelBubble = true;
            if (this.isDrawing) {
                    const originalCoord = imageCoords[vertexIndex];
                    await this.addPoint(originalCoord[0], originalCoord[1]);
                } else if (this.isErasing) {
                    if (this.selectedGroup === group) {
                        this.eraseVertex(group, vertexIndex);
                    } else {
                        this.log('Please select the polygon to erase from.', 'warning');
                    }
                }
            });

            group.add(vertexHandle);
        });

        group.on('mouseenter', () => {
            if (this.isDrawing || this.isErasing) return;
            this.stage.container().style.cursor = 'pointer';
            if (this.selectedGroup !== group) {
                polygon.to({ fill: `rgba(${color}, 0.5)`, duration: 0.1 });
            }
        });

        group.on('mouseleave', () => {
            const cursor = this.isDrawing ? 'crosshair' : (this.isErasing ? 'crosshair' : 'grab');
            this.stage.container().style.cursor = cursor;
            if (this.selectedGroup !== group) {
                polygon.to({ fill: `rgba(${color}, 0.3)`, duration: 0.1 });
            }
        });

        group.on('click tap', (e) => {
            if (this.isDrawing || this.isErasing) return;

            if (e.evt.altKey && this.selectedGroup === group && e.target.className === 'Line') {
                e.cancelBubble = true;
                this.addVertexToPolygon(group, this.stage.getPointerPosition());
                return;
            }

            e.cancelBubble = true;

            if (this.selectedGroup && this.selectedGroup !== group) {
                this.setPolygonSelectionState(this.selectedGroup, false);
            }

            const isNowSelected = this.selectedGroup !== group;
            this.setPolygonSelectionState(group, isNowSelected);
            this.selectedGroup = isNowSelected ? group : null;

            if (isNowSelected) this.log(`Selected area: ${name}. Alt-click an edge to add a point.`);
            else this.log(`Deselected area: ${name}`);

            this.updateToolState();
            this.layer.batchDraw();
        });

        this.layer.add(group);
        this.polygons.push(group);
    }

    getColorForIndex(index) {
        const colors = ['31, 184, 198', '255, 193, 133', '180, 65, 60', '93, 135, 143', '219, 69, 69', '210, 186, 76'];
        return colors[index % colors.length];
    }

    async createStageFromVideoStream() {
        // Capture a token at the start; any later calls that change the sequence will cancel this run
        const creationToken = ++this.stageCreationSeq;
        const streamUrlInput = document.getElementById('streamUrlInput');
        const streamUrl = streamUrlInput?.value?.trim();
        
        if (!streamUrl) {
            this.log('No stream URL provided. Please enter a video stream URL first.', 'warning');
            return false;
        }

        try {
            // Create a temporary video element to get dimensions
            const tempVideo = document.createElement('video');
            tempVideo.crossOrigin = 'anonymous';
            tempVideo.muted = true;
            
            return new Promise((resolve) => {
                tempVideo.onloadedmetadata = () => {
                    if (creationToken !== this.stageCreationSeq) { tempVideo.remove(); return resolve(false); }
                    const videoWidth = tempVideo.videoWidth || 1920;
                    const videoHeight = tempVideo.videoHeight || 1080;
                    
                    this.log(`Video dimensions: ${videoWidth}x${videoHeight}`, 'info');
                    
                    // Create mock canvas with video dimensions
                    if (creationToken === this.stageCreationSeq) {
                        this.createMockCanvas(videoWidth, videoHeight);
                    }
                    
                    // Clean up
                    tempVideo.remove();
                    resolve(true);
                };
                
                tempVideo.onerror = () => {
                    if (creationToken !== this.stageCreationSeq) { tempVideo.remove(); return resolve(false); }
                    this.log('Failed to load video metadata. Using default dimensions.', 'warning');
                    // Use default dimensions if video can't be loaded
                    if (creationToken === this.stageCreationSeq) {
                        this.createMockCanvas(1920, 1080);
                    }
                    tempVideo.remove();
                    resolve(true);
                };
                
                // Set a timeout to avoid hanging
                setTimeout(() => {
                    if (creationToken !== this.stageCreationSeq) { tempVideo.remove(); return resolve(false); }
                    this.log('Video metadata loading timeout. Using default dimensions.', 'warning');
                    if (creationToken === this.stageCreationSeq) {
                        this.createMockCanvas(1920, 1080);
                    }
                    tempVideo.remove();
                    resolve(true);
                }, 5000);
                
                tempVideo.src = streamUrl;
            });
        } catch (error) {
            this.log(`Error creating stage from video stream: ${error.message}`, 'error');
            return false;
        }
    }

    createMockCanvas(videoWidth, videoHeight) {
        const container = document.getElementById('konvaContainer');
        if (!container) {
            this.log('Canvas container not found.', 'error');
            return;
        }

        // Set container size
        const containerWidth = container.clientWidth || 960;
        const containerHeight = container.clientHeight || 540;
        
        // Create stage
        this.stage = new Konva.Stage({ 
            container: 'konvaContainer', 
            width: containerWidth, 
            height: containerHeight, 
            draggable: false 
        });
        
        this.layer = new Konva.Layer();
        this.stage.add(this.layer);
        
        // Create snap indicator
        this.snapIndicator = new Konva.Circle({ 
            x: 0, 
            y: 0, 
            radius: 8, 
            fill: 'rgba(50,184,198,0.5)', 
            visible: false 
        });
        this.layer.add(this.snapIndicator);

        // Create mock background image (rectangle representing video frame)
        const scale = Math.min(containerWidth / videoWidth, containerHeight / videoHeight);
        const displayWidth = videoWidth * scale;
        const displayHeight = videoHeight * scale;
        const offsetX = (containerWidth - displayWidth) / 2;
        const offsetY = (containerHeight - displayHeight) / 2;

        const mockBackground = new Konva.Rect({
            x: offsetX,
            y: offsetY,
            width: displayWidth,
            height: displayHeight,
            fill: '#2a2a2a',
            stroke: '#555',
            strokeWidth: 2,
            listening: false
        });

        // Add grid pattern to make it look more like a video frame
        const gridSize = 50;
        const gridLines = [];
        
        for (let x = offsetX; x <= offsetX + displayWidth; x += gridSize) {
            gridLines.push(new Konva.Line({
                points: [x, offsetY, x, offsetY + displayHeight],
                stroke: '#444',
                strokeWidth: 1,
                listening: false
            }));
        }
        
        for (let y = offsetY; y <= offsetY + displayHeight; y += gridSize) {
            gridLines.push(new Konva.Line({
                points: [offsetX, y, offsetX + displayWidth, y],
                stroke: '#444',
                strokeWidth: 1,
                listening: false
            }));
        }

        this.imageLayer = new Konva.Layer();
        this.stage.add(this.imageLayer);
        this.imageLayer.add(mockBackground);
        gridLines.forEach(line => this.imageLayer.add(line));
        this.imageLayer.moveToBottom();
        this.imageLayer.draw();

        // Create a mock background image object for coordinate conversion
        this.backgroundImage = {
            x: () => offsetX,
            y: () => offsetY,
            width: () => displayWidth,
            height: () => displayHeight,
            image: () => ({
                naturalWidth: videoWidth,
                naturalHeight: videoHeight
            })
        };

        // Show canvas container
        document.getElementById('canvasContainer')?.classList.remove('hidden');
        
        // Draw existing polygons if any
        this.drawExistingPolygons();
        
        // Add stage click listener for drawing
        this.stage.on('click tap', async (e) => {
            if (!this.isDrawing) return;
            if (e.target === this.stage || e.target === mockBackground || gridLines.includes(e.target)) {
                const pos = this.stage.getPointerPosition();
                const imageCoords = this.getCanvasToImageCoordinates(pos.x, pos.y);
                await this.addPoint(imageCoords.x, imageCoords.y);
            }
        });
        
        this.log(`Created mock canvas with video dimensions: ${videoWidth}x${videoHeight}`, 'success');
    }

    closeDrawingCanvas() {
        // Hide canvas container
        document.getElementById('canvasContainer')?.classList.add('hidden');
        
        // Show video stream area if it exists
        const videoStreamArea = document.getElementById('videoStreamArea');
        if (videoStreamArea) {
            videoStreamArea.classList.remove('hidden');
        }
        
        // Invalidate any in-flight stage creation to avoid reopening after close
        this.stageCreationSeq++;

        // Clean up Konva stage and related objects
        if (this.stage) {
            // Detach all listeners to avoid leaked handlers keeping canvas interactive
            try { this.stage.off(); } catch (e) { /* no-op */ }
            this.stage.destroy();
            this.stage = null;
            this.layer = null;
            this.imageLayer = null;
            this.backgroundImage = null;
            this.snapIndicator = null;
        }
        
        // Hard-reset Konva container to avoid stale canvases blocking re-open
        const konvaContainer = document.getElementById('konvaContainer');
        if (konvaContainer) {
            // Remove any leftover canvas elements Konva might have left behind
            while (konvaContainer.firstChild) {
                konvaContainer.removeChild(konvaContainer.firstChild);
            }
        }
        
        // Reset drawing state
        this.isDrawing = false;
        this.currentPolygon = null;
        this.currentArea = null;
        this.drawingAreaType = null;
        this.selectedGroup = null;
        this.polygons = [];
        
        // Remove active states from buttons
        document.getElementById('addArea')?.classList.remove('tool-active');
        document.getElementById('eraserTool')?.classList.remove('tool-active');
        
        // Update drawing status
        this.updateDrawingStatus('Drawing canvas closed. Ready to draw.');
        
        this.log('Drawing canvas closed. Returned to video stream view.', 'success');
    }

    async saveConfigToBackend() {
        try {
            const backendUrlInput = document.getElementById('backendUrlInput');
            const backendUrl = backendUrlInput?.value?.trim();
            
            if (!this.isValidUrl(backendUrl)) {
                this.log('No valid backend URL provided. Configuration saved locally only.', 'warning');
                return;
            }

            const normalized = this.normalizeToBaseConfig(this.config);
            const response = await fetch(`${backendUrl}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ yaml: this.formatYamlWithQuotes(normalized) })
            });

            if (!response.ok) {
                throw new Error(await response.text());
            }

            this.log('Configuration saved to backend successfully.', 'success');
        } catch (error) {
            this.log(`Failed to save configuration to backend: ${error.message}`, 'error');
        }
    }

    async startNewArea(areaType = 'human_counter') {
        // Chỉ cho phép tạo area cho các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot create area for module "${areaType}" - not allowed. Only these modules are permitted: ${this.allowedModules.join(', ')}`, 'warning');
            return;
        }

        // Yêu cầu phải bật stream trước khi vẽ và dựng canvas từ khung hình stream hiện tại
        const liveVideo = document.getElementById('liveStream');
        const hasActiveStream = !!(liveVideo && liveVideo.srcObject && liveVideo.srcObject.getTracks().some(t => t.readyState === 'live'));
        if (!hasActiveStream) {
            this.log('Please start the stream before drawing.', 'warning');
            return;
        }
        await this.ensureKonvaStageFromVideo(liveVideo);
        if (this.isErasing) this.toggleEraser(false);
        if (this.isDrawing) this.cancelCurrentPolygon();

        if (this.selectedGroup) {
            this.setPolygonSelectionState(this.selectedGroup, false);
            this.selectedGroup = null;
            this.updateToolState();
        }

        document.getElementById('addArea').classList.add('tool-active');

        this.drawingAreaType = areaType;
        let areaName = '';

        if (areaType === 'human_counter') {
            areaName = 'Human Counter Area';
            // Ensure module config is initialized even when empty
            if (!this.config.modules) this.config.modules = {};
            if (!this.config.modules.human_counter) this.config.modules.human_counter = {};
            const modObj = this.config.modules.human_counter;
            if (!Array.isArray(modObj.coordinates)) modObj.coordinates = [];

            if (modObj.coordinates && modObj.coordinates.length > 0) {
                if (!confirm('This will replace the existing area. Continue?')) {
                    document.getElementById('addArea').classList.remove('tool-active');
                    return;
                }
                this._syncCoordinates([]);
                this.drawExistingPolygons();
                this.generateConfigForm();
                this.saveState();
                this.saveSessionDebounced();
                this.saveConfigToBackend();
            }
            this.currentArea = { coordinates: [] };
        }

        this.currentPolygon = { points: [], line: null, circles: [] };
        this.isDrawing = true;
        this.stage.container().style.cursor = 'crosshair';

        this.polygons.forEach(g => g.find('.vertex-handle').forEach(h => h.visible(true).listening(true).radius(5 / this.stage.scaleX())));
        if (this.layer) this.layer.batchDraw();

        this.updateDrawingStatus(`Drawing "${areaName}". Click on the image to add points. Click first point to close.`);
        this.log(`Started drawing new area: ${areaName}`);
    }

    async addPoint(x, y) {
        if (!this.isDrawing || !this.currentPolygon) return;

        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(this.drawingAreaType)) {
            this.log(`Cannot add point for module "${this.drawingAreaType}" - not allowed.`, 'warning');
            return;
        }

        if (this.currentPolygon.points.length > 2) {
            const firstPoint = this.currentPolygon.points[0];
            const firstPointCanvas = this.getImageToCanvasCoordinates(firstPoint[0], firstPoint[1]);
            const currentPointCanvas = this.getImageToCanvasCoordinates(x, y);
            const distance = Math.hypot(currentPointCanvas.x - firstPointCanvas.x, currentPointCanvas.y - firstPointCanvas.y);
            if (distance < (15 / this.stage.scaleX())) {
                await this.finishPolygon();
                return;
            }
        }

        // Store coordinates normalized to image dimensions (0..1)
        let normPoint = [x, y];
        if (this.backgroundImage && this.backgroundImage.image) {
            const img = this.backgroundImage.image();
            const imgW = img.naturalWidth;
            const imgH = img.naturalHeight;
            if (imgW > 0 && imgH > 0) {
                normPoint = [x / imgW, y / imgH];
            }
        }

        this.currentPolygon.points.push([x, y]);
        this.currentArea.coordinates.push(normPoint);

        this.drawCurrentPolygon();
        this.updateDrawingStatus(`Added point ${this.currentPolygon.points.length}. ${this.currentPolygon.points.length < 3 ? 'Add more points or' : ''} Click first point to close polygon.`);
    }

    drawCurrentPolygon() {
        if (!this.currentPolygon || !this.layer) return;

        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(this.drawingAreaType)) {
            this.log(`Cannot draw polygon for module "${this.drawingAreaType}" - not allowed.`, 'warning');
            return;
        }

        if (this.currentPolygon.line) this.currentPolygon.line.destroy();
        this.currentPolygon.circles.forEach(circle => circle.destroy());
        this.currentPolygon.circles = [];

        if (this.currentPolygon.points.length > 0) {
            const canvasPoints = this.currentPolygon.points.map(p => this.getImageToCanvasCoordinates(p[0], p[1]));
            const lineColor = 'rgb(31, 184, 198)';
            const pointColor = lineColor;
            const firstPointColor = 'rgb(219, 69, 69)';

            this.currentPolygon.line = new Konva.Line({
                points: canvasPoints.flatMap(p => [p.x, p.y]),
                stroke: lineColor, strokeWidth: 2, closed: false
            });
            this.layer.add(this.currentPolygon.line);

            canvasPoints.forEach((point, index) => {
                const circle = new Konva.Circle({
                    x: point.x, y: point.y,
                    radius: (index === 0 ? 8 : 5) / this.stage.scaleX(),
                    fill: index === 0 ? firstPointColor : pointColor,
                    stroke: 'white', strokeWidth: 2 / this.stage.scaleX()
                });

                if (index === 0 && this.currentPolygon.points.length > 2) {
                    circle.on('mouseenter', () => {
                        this.stage.container().style.cursor = 'pointer';
                        circle.to({ radius: 12 / this.stage.scaleX(), fill: 'rgb(255, 0, 0)', duration: 0.1 });
                    });
                    circle.on('mouseleave', () => {
                        this.stage.container().style.cursor = 'crosshair';
                        circle.to({ radius: 8 / this.stage.scaleX(), fill: firstPointColor, duration: 0.1 });
                    });
                    circle.on('click tap', async (e) => {
                        e.cancelBubble = true;
                        if (this.isDrawing) await this.finishPolygon();
                    });
                }
                this.currentPolygon.circles.push(circle);
                this.layer.add(circle);
            });
            this.layer.draw();
        }
    }

    async finishPolygon() {
        if (!this.isDrawing || !this.currentPolygon || this.currentPolygon.points.length < 3) return;

        const areaType = this.drawingAreaType;
        
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot finish polygon for module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        let areaName = '';
        if (this.currentPolygon.line) this.currentPolygon.line.destroy();
        this.currentPolygon.circles.forEach(circle => circle.destroy());

        if (areaType === 'human_counter') {
            areaName = 'Human Counter Area';
            // Sync coordinates across System, Module and Human Counter
            this._syncCoordinates(this.currentArea.coordinates);
        }

        // Reset drawing state immediately so UI reflects completion regardless of save outcome
        this.isDrawing = false;
        this.currentPolygon = null;
        this.currentArea = null;
        this.drawingAreaType = null;
        document.getElementById('addArea')?.classList.remove('tool-active');
        if (this.stage) this.stage.container().style.cursor = 'grab';

        this.drawExistingPolygons();
        this.updateDrawingStatus(`✅ Polygon "${areaName}" completed successfully! Canvas will close automatically...`);
        this.generateConfigForm();
        this.saveState();
        this.saveSessionDebounced();

        try {
            // Save to backend YAML file (non-blocking for closing canvas)
            await this.saveConfigToBackend();
            this.log(`✅ Completed area: ${areaName} - Keypoints saved successfully!`);
        } catch (e) {
            // Should already be handled inside saveConfigToBackend, but keep guard
            this.log(`Save after drawing failed: ${e?.message || e}`, 'warning');
        } finally {
            // Ensure canvas is closed even if saving fails
            setTimeout(() => {
                this.closeDrawingCanvas();
                this.log('🎯 Drawing canvas closed automatically. Area saved and ready for use!', 'success');
                // Reconnect stream to apply new polygon configuration
                const canReconnect = this.isValidUrl(this.streamUrl || '') && this.isValidUrl(this.backendUrl || '');
                if (canReconnect) {
                    (async () => {
                        try {
                            await this.stopStream();
                        } catch (_) { /* no-op */ }
                        try {
                            await this.startWebRTCStream(this.streamUrl, this.backendUrl);
                            this.log('Stream reconnected with updated polygon.', 'success');
                        } catch (err) {
                            this.log(`Failed to reconnect stream: ${err?.message || err}`, 'error');
                        }
                    })();
                } else {
                    this.log('Skipping stream reconnect (missing stream/backend URL).', 'info');
                }
            }, 1200);
        }
    }

    cancelCurrentPolygon() {
        if (this.currentPolygon) {
            if (this.currentPolygon.line) this.currentPolygon.line.destroy();
            this.currentPolygon.circles.forEach(circle => circle.destroy());
        }

        this.isDrawing = false;
        this.currentPolygon = null;
        this.currentArea = null;
        this.drawingAreaType = null;
        document.getElementById('addArea').classList.remove('tool-active');
        if (this.stage) this.stage.container().style.cursor = 'grab';

        this.drawExistingPolygons();
        if (this.layer) this.layer.draw();

        this.updateDrawingStatus('Drawing cancelled. Ready to draw.');
    }

    updateZoomDisplay() {
        const zoomLevel = document.getElementById('zoomLevel');
        if (zoomLevel) zoomLevel.textContent = `${Math.round(this.zoom * 100)}%`;
    }

    updateDrawingStatus(message) {
        const drawingStatus = document.getElementById('drawingStatus');
        if (drawingStatus) drawingStatus.textContent = message;
    }

    updateToolState() {
        const isAreaSelected = !!this.selectedGroup;
        const deleteBtn = document.getElementById('deleteSelected');
        const frontBtn = document.getElementById('bringToFront');
        const backBtn = document.getElementById('sendToBack');
        const eraserBtn = document.getElementById('eraserTool');

        if (deleteBtn) deleteBtn.disabled = !isAreaSelected;
        if (frontBtn) frontBtn.disabled = !isAreaSelected;
        if (backBtn) backBtn.disabled = !isAreaSelected;
        if (eraserBtn) eraserBtn.disabled = !isAreaSelected;

        if (!isAreaSelected && this.isErasing) this.toggleEraser(false);
    }

    toggleVertexHandles(visible) {
        this.polygons.forEach(group => {
            const areaType = group.getAttr('areaType');
            // Chỉ xử lý các module được phép
            if (this.allowedModules.includes(areaType)) {
                group.find('.vertex-handle').forEach(handle => handle.visible(visible));
            }
        });
        if (this.layer) this.layer.batchDraw();
    }

    toggleAreaVisibility(areaType, areaIndex) {
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot toggle visibility for module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        const identifier = `${areaType}_${areaIndex}`;
        const group = this.polygons.find(p => p.getAttr('areaType') === areaType && p.getAttr('areaIndex') === areaIndex);
        if (group) {
            const isVisible = group.visible();
            group.visible(!isVisible);
            if (isVisible) this.hiddenAreaIdentifiers.add(identifier);
            else this.hiddenAreaIdentifiers.delete(identifier);
            this.layer.batchDraw();
            this.log(`Toggled visibility for area ${areaType} index ${areaIndex}`);

            const button = document.querySelector(`.toggle-visibility[data-type="${areaType}"][data-index="${areaIndex}"]`);
            if (button) {
                const isNowVisible = !this.hiddenAreaIdentifiers.has(identifier);
                button.innerHTML = isNowVisible ? '👁️' : '🙈';
                button.title = isNowVisible ? 'Hide Area' : 'Show Area';
            }
        }
    }

    bringToFront() {
        if (!this.selectedGroup) return;
        const areaType = this.selectedGroup.getAttr('areaType');
        const areaIndex = this.selectedGroup.getAttr('areaIndex');

        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot bring to front module "${areaType}" - not allowed.`, 'warning');
            return;
        }
    }

    sendToBack() {
        if (!this.selectedGroup) return;
        const areaType = this.selectedGroup.getAttr('areaType');
        const areaIndex = this.selectedGroup.getAttr('areaIndex');

        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot send to back module "${areaType}" - not allowed.`, 'warning');
            return;
        }
    }

    updateAreasAfterReorder() {
        this.drawExistingPolygons();
        this.generateConfigForm();
        this.saveSessionDebounced();
    }

    generateConfigForm() {
        const container = document.getElementById('configForm');
        if (!container) return;

        const expandedKeys = new Set();
        container.querySelectorAll('.collapsible-section').forEach(section => {
            if (section.querySelector('.section-content.expanded')) expandedKeys.add(section.dataset.key);
        });

        container.innerHTML = '';
        Object.keys(this.config).forEach(key => {
            const title = `${key.charAt(0).toUpperCase() + key.slice(1)} Configuration`;
            const sectionDiv = this.createCollapsibleSection(title, key, expandedKeys);
            this.generateFormFields(this.config[key], sectionDiv.querySelector('.section-content'), key, expandedKeys);
            container.appendChild(sectionDiv);
        });
    }

    initializeSettingsNavigation() {
        // Setup navigation between settings sections
        const navLinks = document.querySelectorAll('.settings-nav-link');
        const sections = document.querySelectorAll('.settings-section');

        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetSection = link.dataset.section;
                
                // Update active nav link
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
                
                // Show target section
                sections.forEach(section => {
                    section.classList.remove('active');
                    if (section.id === `${targetSection}-section`) {
                        section.classList.add('active');
                    }
                });
                
                // Generate specific config forms for each section
                this.generateSectionConfig(targetSection);
            });
        });
    }

    generateSectionConfig(sectionName) {
        let containerId, configData;
        
        switch(sectionName) {
            case 'general':
                containerId = 'configForm';
                configData = this.config;
                break;
            case 'modules':
                containerId = 'modulesConfigForm';
                // Chỉ hiển thị các module được phép
                const filteredModules = {};
                this.allowedModules.forEach(moduleKey => {
                    if (this.config.modules[moduleKey]) {
                        filteredModules[moduleKey] = this.config.modules[moduleKey];
                    }
                });
                configData = { modules: filteredModules };
                break;
            case 'pipeline':
                containerId = 'pipelineConfigForm';
                configData = { pipeline: this.config.pipeline };
                break;
            case 'logging':
                containerId = 'loggingConfigForm';
                configData = { logging: this.config.logging };
                break;
            case 'database':
                containerId = 'databaseConfigForm';
                configData = { database: this.config.database };
                break;
            case 'advanced':
                containerId = 'advancedConfigForm';
                configData = { application: this.config.application };
                break;
            default:
                return;
        }

        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = '';
        Object.keys(configData).forEach(key => {
            const title = `${key.charAt(0).toUpperCase() + key.slice(1)} Configuration`;
            const sectionDiv = this.createCollapsibleSection(title, key, new Set());
            this.generateFormFields(configData[key], sectionDiv.querySelector('.section-content'), key, new Set());
            container.appendChild(sectionDiv);
        });
    }

    createCollapsibleSection(title, key, expandedKeys = new Set()) {
        const section = document.createElement('div');
        section.className = 'collapsible-section';
        section.dataset.key = key;

        const header = document.createElement('div');
        header.className = 'section-header';
        header.innerHTML = `<span>${title}</span><svg class="chevron" fill="none" stroke="currentColor" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"></polyline></svg>`;

        const content = document.createElement('div');
        content.className = 'section-content';

        header.addEventListener('click', () => {
            const isExpanded = content.classList.contains('expanded');
            content.classList.toggle('expanded', !isExpanded);
            header.querySelector('.chevron').classList.toggle('rotated', !isExpanded);
        });

        if (expandedKeys.has(key)) {
            content.classList.add('expanded');
            header.querySelector('.chevron').classList.add('rotated');
        }

        section.appendChild(header);
        section.appendChild(content);
        return section;
    }

    generateFormFields(obj, container, parentKey, expandedKeys = new Set()) {
        if (parentKey === 'modules') {
            container.innerHTML = '';
            // Chỉ hiển thị các module được phép
            this.allowedModules.forEach(moduleKey => {
                if (obj[moduleKey]) {
                    const moduleSection = this._createModuleConfigSection(moduleKey, expandedKeys);
                    if (moduleSection) container.appendChild(moduleSection);
                }
            });
            return;
        }

        Object.keys(obj).forEach(key => {
            const fullKey = `${parentKey}.${key}`;
            const value = obj[key];
            if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                const subsection = this.createCollapsibleSection(key.replace(/_/g, ' ').toUpperCase(), fullKey, expandedKeys);
                this.generateFormFields(value, subsection.querySelector('.section-content'), fullKey, expandedKeys);
                container.appendChild(subsection);
            } else {
                container.appendChild(this.createFormField(key, value, fullKey));
            }
        });
    }

    _createModuleConfigSection(moduleKey, expandedKeys = new Set()) {
        // Chỉ tạo section cho các module được phép
        if (!this.allowedModules.includes(moduleKey)) return null;
        
        const obj = this.config.modules;
        if (!obj[moduleKey]) return null;

        const fullKey = `modules.${moduleKey}`;
        if (this.configMode === 'simple' && this.advancedKeys.has(fullKey)) return null;

        const subsection = this.createCollapsibleSection(moduleKey.replace(/_/g, ' ').toUpperCase(), fullKey, expandedKeys);
        const contentDiv = subsection.querySelector('.section-content');
        this.generateFormFields(obj[moduleKey], contentDiv, fullKey, expandedKeys);

        const isHumanCounter = moduleKey === 'human_counter';
        if (isHumanCounter && obj[moduleKey].visualize === true) {
            const wrapper = document.createElement('div');
            wrapper.style.cssText = 'margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--color-border); display: flex; gap: 8px; align-items: center;';

            const color = '31, 184, 198';
            const indicator = document.createElement('span');
            indicator.className = 'color-indicator';
            indicator.style.backgroundColor = `rgb(${color})`;

            const drawButton = document.createElement('button');
            drawButton.textContent = (obj[moduleKey].coordinates && obj[moduleKey].coordinates.length > 0) ? 'Edit Area' : 'Draw Area';
            drawButton.className = 'btn btn--secondary btn--sm';
            drawButton.onclick = async () => {
                // Close settings modal first to show the drawing canvas
                const settingsModal = document.getElementById('settingsModal');
                if (settingsModal) {
                    settingsModal.classList.add('hidden');
                }
                // Start drawing area
                await this.startNewArea('human_counter');
            };

            // Restore Clear Area button to allow removing existing polygon
            const clearButton = document.createElement('button');
            clearButton.textContent = 'Clear Area';
            clearButton.className = 'btn btn--outline btn--sm';
            // Always keep the Clear Area button enabled and visible
            clearButton.disabled = false;
            clearButton.onclick = async () => {
                const currentCoords = (this.config.modules?.human_counter?.coordinates) || [];
                if (!confirm('Clear the current area polygon?')) return;
                // Reset coordinates for human_counter and refresh drawings/UI
                this._syncCoordinates([]);
                this.drawExistingPolygons();
                this.saveState();
                this.saveSessionDebounced();
                await this.saveConfigToBackend();
                // Close settings modal after clearing polygon
                const settingsModal = document.getElementById('settingsModal');
                if (settingsModal) {
                    settingsModal.classList.add('hidden');
                }
                // Regenerate to update button states/labels
                this.generateConfigForm();
                this.log('Cleared Human Counter area polygon.', 'info');
                // If streaming, restart to apply cleared polygon immediately
                const canReconnect = this.isValidUrl(this.streamUrl || '') && this.isValidUrl(this.backendUrl || '');
                if (canReconnect) {
                    try { await this.stopStream(); } catch (_) { /* no-op */ }
                    try {
                        await this.startWebRTCStream(this.streamUrl, this.backendUrl);
                        this.log('Stream reconnected after clearing polygon.', 'success');
                    } catch (err) {
                        this.log(`Failed to reconnect stream: ${err?.message || err}`, 'error');
                    }
                }
            };

            wrapper.append(indicator, drawButton, clearButton);
            contentDiv.appendChild(wrapper);
        }
        return subsection;
    }

    createPipelineModulesControl() {
        const wrapper = document.createElement('div');
        wrapper.className = 'form-row-vertical';
        wrapper.innerHTML = `<label class="field-label">MODULES</label>`;
        const activeModulesContainer = document.createElement('div');
        activeModulesContainer.className = 'active-modules-list';
        const addModuleContainer = document.createElement('div');
        addModuleContainer.className = 'add-module-container';
        const select = document.createElement('select');
        select.className = 'field-input';
        const addButton = document.createElement('button');
        addButton.type = 'button';
        addButton.textContent = 'Add Module';
        addButton.className = 'btn btn--secondary btn--sm';
        addButton.onclick = () => select.value && this.addPipelineModule(select.value);

        const updateUI = () => {
            activeModulesContainer.innerHTML = '';
            (this.config.pipeline.modules || []).forEach(moduleName => {
                const chip = document.createElement('div');
                chip.className = 'module-chip';
                chip.textContent = moduleName;
                const removeBtn = document.createElement('button');
                removeBtn.innerHTML = '&times;';
                removeBtn.className = 'remove-module-btn';
                removeBtn.onclick = () => this.removePipelineModule(moduleName);
                chip.appendChild(removeBtn);
                activeModulesContainer.appendChild(chip);
            });

            select.innerHTML = '<option value="">Select a module to add...</option>';
            const available = this.allowedModules.filter(m => !(this.config.pipeline.modules || []).includes(m));
            available.forEach(m => select.add(new Option(m, m)));
        };

        this.updatePipelineModulesUI = updateUI;
        updateUI();

        addModuleContainer.append(select, addButton);
        wrapper.append(activeModulesContainer, addModuleContainer);
        return wrapper;
    }

    addPipelineModule(moduleName) {
        // Chỉ cho phép thêm các module được phép
        if (!this.allowedModules.includes(moduleName)) {
            this.log(`Module "${moduleName}" is not allowed. Only these modules are permitted: ${this.allowedModules.join(', ')}`, 'warning');
            return;
        }

        if (!this.config.pipeline.modules) this.config.pipeline.modules = [];
        if (this.config.pipeline.modules.includes(moduleName)) return;

        this.config.pipeline.modules.push(moduleName);
        if (!this.config.modules[moduleName]) {
            this.config.modules[moduleName] = JSON.parse(JSON.stringify(this.defaultConfig.modules[moduleName]));
        }
        this.log(`Module added: ${moduleName}`, 'info');

        this.updatePipelineModulesUI();

        const modulesContent = document.querySelector('.collapsible-section[data-key="modules"] .section-content');
        if (modulesContent) {
            const newModuleSection = this._createModuleConfigSection(moduleName);
            if (newModuleSection) modulesContent.appendChild(newModuleSection);
            if (modulesContent.querySelector('p')) modulesContent.querySelector('p').remove();
        } else {
            this.generateConfigForm(); // Fallback
        }
        this.saveSessionDebounced();
    }

    removePipelineModule(moduleName) {
        // Chỉ cho phép xóa các module được phép
        if (!this.allowedModules.includes(moduleName)) {
            this.log(`Cannot remove module "${moduleName}" - not allowed.`, 'warning');
            return;
        }

        const index = this.config.pipeline.modules.indexOf(moduleName);
        if (index > -1) {
            this.config.pipeline.modules.splice(index, 1);
            this.log(`Module removed: ${moduleName}`, 'info');
            this.updatePipelineModulesUI();

            const moduleSection = document.querySelector(`.collapsible-section[data-key="modules.${moduleName}"]`);
            if (moduleSection) moduleSection.remove();

            const modulesContent = document.querySelector('.collapsible-section[data-key="modules"] .section-content');
            if (modulesContent && this.config.pipeline.modules.length === 0) {
                modulesContent.innerHTML = `<p style='color: var(--color-text-secondary); font-style: italic; padding: 0 1rem;'>Add modules in "Pipeline Configuration" to see their options here.</p>`;
            }

            this.saveSessionDebounced();
        }
    }


    createFormField(key, value, fullKey) {
        const row = document.createElement('div');
        row.className = 'form-row';
        const label = document.createElement('label');
        label.className = 'field-label';
        label.textContent = key.replace(/_/g, ' ').toUpperCase();
        const input = this.createInputElement(key, value, fullKey);
        row.appendChild(label);
        row.appendChild(input);
        return row;
    }

    createInputElement(key, value, fullKey) {
        if (this.selectOptions[fullKey]) {
            const select = document.createElement('select');
            select.className = 'field-input';
            select.dataset.path = fullKey;
            this.selectOptions[fullKey].forEach(option => {
                const opt = new Option(option, option);
                opt.selected = option === value;
                select.add(opt);
            });
            return select;
        }

        const input = document.createElement('input');
        input.className = 'field-input';
        input.dataset.path = fullKey;

        if (typeof value === 'boolean') {
            input.type = 'checkbox';
            input.className += ' checkbox-input';
            input.checked = value;
            return input;
        }

        if (typeof value === 'number') {
            if (fullKey.includes('confidence') || fullKey.includes('thresh')) {
                input.type = 'range';
                input.className += ' range-input';
                input.min = '0'; input.max = '1'; input.step = '0.01';
                input.value = value;
                const wrapper = document.createElement('div');
                wrapper.style.display = 'flex';
                wrapper.style.alignItems = 'center';
                const valueDisplay = document.createElement('span');
                valueDisplay.textContent = value.toFixed(2);
                valueDisplay.style.marginLeft = '8px';
                valueDisplay.style.fontSize = '12px';
                input.addEventListener('input', () => valueDisplay.textContent = parseFloat(input.value).toFixed(2));
                wrapper.append(input, valueDisplay);
                return wrapper;
            } else {
                input.type = 'number';
                input.value = value;
            }
        } else if (Array.isArray(value)) {
            input.value = JSON.stringify(value);
        } else {
            input.type = 'text';
            input.value = value || '';
        }

        return input;
    }

    updateConfigFromForm() {
        // Get all form containers
        const formContainers = [
            'configForm', 'modulesConfigForm', 'pipelineConfigForm', 
            'loggingConfigForm', 'databaseConfigForm', 'advancedConfigForm'
        ];
        
        formContainers.forEach(containerId => {
            const container = document.getElementById(containerId);
            if (!container) return;
            
            const inputs = container.querySelectorAll('.field-input[data-path]');
            inputs.forEach(input => {
                const path = input.dataset.path;
                const keys = path.split('.');
                
                // Kiểm tra xem có phải là module không được phép không
                if (keys[0] === 'modules' && keys.length > 1) {
                    const moduleKey = keys[1];
                    if (!this.allowedModules.includes(moduleKey)) {
                        return; // Bỏ qua các module không được phép
                    }
                }
                
                let current = this.config;
                for (let i = 0; i < keys.length - 1; i++) {
                    if (!current[keys[i]]) current[keys[i]] = {};
                    current = current[keys[i]];
                }
                const lastKey = keys[keys.length - 1];
                let value = input.type === 'checkbox' ? input.checked : input.value;
                if (input.type === 'number' || input.type === 'range') value = parseFloat(value);
                else if (typeof value === 'string' && value.startsWith('[') && value.endsWith(']')) {
                    try { value = JSON.parse(value); } catch (e) { /* ignore */ }
                }
                current[lastKey] = value;
            });
        });
    }


    saveSession() {
        try {
            // Lọc bỏ các module không được phép trước khi lưu
            const filteredConfig = JSON.parse(JSON.stringify(this.config));
            if (filteredConfig.modules) {
                const filteredModules = {};
                this.allowedModules.forEach(moduleKey => {
                    if (filteredConfig.modules[moduleKey]) {
                        filteredModules[moduleKey] = filteredConfig.modules[moduleKey];
                    }
                });
                filteredConfig.modules = filteredModules;
            }
            
            // Lọc pipeline modules
            if (filteredConfig.pipeline && filteredConfig.pipeline.modules) {
                filteredConfig.pipeline.modules = filteredConfig.pipeline.modules.filter(module => this.allowedModules.includes(module));
            }

            localStorage.setItem('smartcity_session', JSON.stringify({
                config: filteredConfig,
                configMode: this.configMode,
                timestamp: Date.now(),
                currentVideoProcessingMode: this.currentVideoProcessingMode
            }));
            const sessionStatus = document.getElementById('sessionStatus');
            if (sessionStatus) {
                sessionStatus.title = 'Session Saved';
                sessionStatus.className = 'btn btn-icon btn--outline btn--sm session-saved';
            }
        } catch (error) {
            this.log(`Session save failed: ${error.message}`, 'error');
        }
    }

    loadSession() {
        try {
            const sessionData = localStorage.getItem('smartcity_session');
            if (sessionData) {
                const parsedData = JSON.parse(sessionData);
                this.config = JSON.parse(JSON.stringify(this.defaultConfig)) || parsedData.config;
                
                // Lọc bỏ các module không được phép
                if (this.config.modules) {
                    const filteredModules = {};
                    this.allowedModules.forEach(moduleKey => {
                        if (this.config.modules[moduleKey]) {
                            filteredModules[moduleKey] = this.config.modules[moduleKey];
                        }
                    });
                    this.config.modules = filteredModules;
                }
                
                // Lọc pipeline modules
                if (this.config.pipeline && this.config.pipeline.modules) {
                    this.config.pipeline.modules = this.config.pipeline.modules.filter(module => this.allowedModules.includes(module));
                }
                
                this.configMode = parsedData.configMode || 'simple';
                this.currentVideoProcessingMode = parsedData.currentVideoProcessingMode || 'upload';
                const modeUploadFile = document.getElementById('modeUploadFile');
                const modeStream = document.getElementById('modeStream');
                if (this.currentVideoProcessingMode === 'upload') {
                    if (modeUploadFile) modeUploadFile.checked = true;
                } else {
                    if (modeStream) modeStream.checked = true;
                }

                this.log('Session restored', 'success');
                const sessionStatus = document.getElementById('sessionStatus');
                if (sessionStatus) {
                    sessionStatus.title = 'Session Restored';
                    sessionStatus.className = 'btn btn-icon btn--outline btn--sm session-restored';
                }
            } else {
                this.config = JSON.parse(JSON.stringify(this.defaultConfig));
                this.configMode = 'simple';
                this.currentVideoProcessingMode = 'upload';
            }
        } catch (error) {
            this.log('Failed to restore session', 'error');
            this.config = JSON.parse(JSON.stringify(this.defaultConfig));
            this.configMode = 'simple';
            this.currentVideoProcessingMode = 'upload';
        }
    }

    initializeSessionStatus() {
        const sessionStatus = document.getElementById('sessionStatus');
        if (sessionStatus) {
            // Thiết lập trạng thái ban đầu nếu chưa có session data
            if (!localStorage.getItem('smartcity_session')) {
                sessionStatus.title = 'Session Active';
                sessionStatus.className = 'btn btn-icon btn--outline btn--sm session-active';
            }
        }
    }

    showLargeVideoPlayer() {
        if (!this.renderedVideoUrl) {
            this.log('No rendered video available to view.', 'warning');
            return;
        }
        const smallVideo = document.getElementById('smallResultVideo');
        if (smallVideo && !smallVideo.paused) smallVideo.pause();

        document.querySelector('.canvas-section').classList.add('hidden');
        document.getElementById('largeVideoPlayer').classList.remove('hidden');

        const largeVideo = document.getElementById('largeResultVideo');
        if (largeVideo) largeVideo.play().catch(e => this.log(`Autoplay prevented in large player: ${e.message}`, 'info'));
        this.log('Showing rendered video in main panel.');
    }

    hideLargeVideoPlayer() {
        document.getElementById('largeVideoPlayer').classList.add('hidden');
        document.querySelector('.canvas-section').classList.remove('hidden');
        const largeVideo = document.getElementById('largeResultVideo');
        if (largeVideo) largeVideo.pause();
        this.log('Closed large video viewer, returning to editor.');
    }

    log(message, type = 'info') {
        try {
            const entry = document.createElement('div');
            entry.className = `log-entry ${type}`;
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            const container = document.getElementById('logsContainer');
            if (container) {
                container.appendChild(entry);
                container.scrollTop = container.scrollHeight;
            }
            console.log(`[SmartCity] ${message}`);
        } catch (error) {
            console.error('Logging failed:', error);
        }
    }

    addVertexToPolygon(group, canvasPos) {
        if (!group) return;

        const line = group.findOne('Line');
        const vertices = [];
        const points = line.points();
        for (let i = 0; i < points.length; i += 2) vertices.push({ x: points[i], y: points[i + 1] });

        if (vertices.length < 2) return;

        let minDistance = Infinity;
        let insertionIndex = -1;

        for (let i = 0; i < vertices.length; i++) {
            const p1 = vertices[i];
            const p2 = vertices[(i + 1) % vertices.length];
            const l2 = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2;
            if (l2 === 0) continue;
            let t = ((canvasPos.x - p1.x) * (p2.x - p1.x) + (canvasPos.y - p1.y) * (p2.y - p1.y)) / l2;
            t = Math.max(0, Math.min(1, t));
            const closestPointOnSegment = { x: p1.x + t * (p2.x - p1.x), y: p1.y + t * (p2.y - p1.y) };
            const dist = Math.hypot(canvasPos.x - closestPointOnSegment.x, canvasPos.y - closestPointOnSegment.y);
            if (dist < minDistance) {
                minDistance = dist;
                insertionIndex = i + 1;
            }
        }

        const newPointImageCoords = this.getCanvasToImageCoordinates(canvasPos.x, canvasPos.y);
        const roundedCoords = [Math.round(newPointImageCoords.x * 10) / 10, Math.round(newPointImageCoords.y * 10) / 10];

        const areaType = group.getAttr('areaType');
        const areaIndex = group.getAttr('areaIndex');
        
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot add vertex to module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        let coordsList, areaName;
        if (areaType === 'human_counter') {
            coordsList = this.config.modules.human_counter.coordinates;
            areaName = 'Human Counter Area';
        }

        if (coordsList) {
            coordsList.splice(insertionIndex, 0, roundedCoords);
            if (areaType === 'human_counter') this._syncCoordinates([...coordsList]);
            this.log(`Added vertex to "${areaName}"`);
            this.drawExistingPolygons();
            const newGroup = this.polygons.find(p => p.getAttr('areaType') === areaType && p.getAttr('areaIndex') === areaIndex);
            if (newGroup) {
                this.selectedGroup = newGroup;
                this.setPolygonSelectionState(newGroup, true);
            }
            this.generateConfigForm();
            this.saveState();
            this.saveSessionDebounced();
            this.saveConfigToBackend();
        }
    }

    setPolygonSelectionState(group, isSelected) {
        const polygon = group.findOne('Line');
        const areaType = group.getAttr('areaType');
        const areaIndex = group.getAttr('areaIndex');
        
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot set selection state for module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        const color = this.getColorForIndex(areaIndex);
        const handles = group.find('.vertex-handle');
        const scale = this.stage.scaleX();

        if (isSelected) {
            polygon.to({ fill: `rgba(${color}, 0.6)`, strokeWidth: 4 / scale, duration: 0.1 });
            handles.forEach(handle => {
                handle.visible(true);
                handle.draggable(true);
            });
        } else {
            polygon.to({ fill: `rgba(${color}, 0.3)`, strokeWidth: 2 / scale, duration: 0.1 });
            handles.forEach(handle => {
                handle.visible(false);
                handle.draggable(false);
            });
        }
    }

    updatePolygonOnVertexDrag(group, draggedVertexIndex) {
        const areaType = group.getAttr('areaType');
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot update polygon for module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        const line = group.findOne('Line');
        const handles = group.find('.vertex-handle');
        const draggedHandle = handles[draggedVertexIndex];
        const scale = this.stage.scaleX();
        let finalPos = { x: draggedHandle.x(), y: draggedHandle.y() };

        const snapDistanceOnCanvas = this.snapThreshold;
        let closestPoint = null;
        let minDistance = Infinity;

        this.polygons.forEach(otherGroup => {
            const otherAreaType = otherGroup.getAttr('areaType');
            // Chỉ xử lý các module được phép
            if (!this.allowedModules.includes(otherAreaType)) {
                return;
            }

            otherGroup.find('.vertex-handle').forEach((otherHandle, otherIndex) => {
                if (otherGroup === group && otherIndex === draggedVertexIndex) return;
                const dist = Math.hypot(finalPos.x - otherHandle.x(), finalPos.y - otherHandle.y());
                if (dist < minDistance) {
                    minDistance = dist;
                    closestPoint = { x: otherHandle.x(), y: otherHandle.y() };
                }
            });
        });

        if (closestPoint && minDistance < snapDistanceOnCanvas) {
            finalPos = closestPoint;
            draggedHandle.position(finalPos);
            this.snapIndicator.position(finalPos);
            this.snapIndicator.radius(8 / scale);
            this.snapIndicator.visible(true);
        } else {
            this.snapIndicator.visible(false);
        }

        const newPoints = handles.map(handle => [handle.x(), handle.y()]).flat();
        line.points(newPoints);

        if (draggedVertexIndex === 0) {
            const label = group.findOne('Text');
            if (label) label.y(draggedHandle.y() - (20 / scale));
        }
        this.layer.batchDraw();
    }

    updateConfigOnVertexDragEnd(group) {
        this.snapIndicator.visible(false);
        const areaType = group.getAttr('areaType');
        const areaIndex = group.getAttr('areaIndex');
        
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot update coordinates for module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        const handles = group.find('.vertex-handle');

        const newCoords = handles.map(handle => {
            const canvasPos = { x: handle.x(), y: handle.y() };
            const imagePos = this.getCanvasToImageCoordinates(canvasPos.x, canvasPos.y);
            // Save normalized coordinates (0..1)
            if (this.backgroundImage && this.backgroundImage.image) {
                const img = this.backgroundImage.image();
                const imgW = img.naturalWidth;
                const imgH = img.naturalHeight;
                if (imgW > 0 && imgH > 0) {
                    return [
                        Math.round((imagePos.x / imgW) * 10000) / 10000,
                        Math.round((imagePos.y / imgH) * 10000) / 10000
                    ];
                }
            }
            return [Math.round(imagePos.x * 10) / 10, Math.round(imagePos.y * 10) / 10];
        });

        if (areaType === 'human_counter') {
            this._syncCoordinates(newCoords);
        }

        this.saveState();
        this.saveSessionDebounced();
        this.log('Updated polygon coordinates.');
        this.generateConfigForm();
        this.saveConfigToBackend();
    }

    updateScaleDependentElements() {
        if (!this.stage) return;
        const scale = this.stage.scaleX();

        this.polygons.forEach(group => {
            const areaType = group.getAttr('areaType');
            // Chỉ xử lý các module được phép
            if (!this.allowedModules.includes(areaType)) {
                return;
            }

            const line = group.findOne('Line');
            const label = group.findOne('Text');
            const handles = group.find('.vertex-handle');
            const isSelected = this.selectedGroup === group;

            if (label) {
                const firstPoint = line.points();
                label.fontSize(14 / scale);
                label.padding(4 / scale);
                if (firstPoint.length >= 2) {
                    label.x(firstPoint[0]);
                    label.y(firstPoint[1] - (20 / scale));
                }
            }
            if (line) line.strokeWidth((isSelected ? 4 : 2) / scale);
            handles.forEach(handle => {
                handle.radius(6 / scale);
                handle.strokeWidth(1 / scale);
            });
        });

        if (this.isDrawing && this.currentPolygon) {
            this.currentPolygon.circles.forEach((circle, index) => {
                circle.radius((index === 0 ? 8 : 5) / scale);
                circle.strokeWidth(2 / scale);
            });
        }
        if (this.snapIndicator) this.snapIndicator.radius(8 / scale);

        this.layer.batchDraw();
    }

    clearHistoryAndSaveInitialState() {
        this.actionHistory = [];
        this.actionIndex = -1;
        this.updateUndoRedoButtons();
    }

    saveState(isInitial = false) {
        const state = {
            human_counter: JSON.parse(JSON.stringify(this.config.modules.human_counter?.coordinates || []))
        };

        if (!isInitial && this.actionHistory.length > 0) {
            const lastState = this.actionHistory[this.actionIndex];
            if (JSON.stringify(lastState) === JSON.stringify(state)) return;
        }

        this.actionHistory = this.actionHistory.slice(0, this.actionIndex + 1);
        this.actionHistory.push(state);
        this.actionIndex = this.actionHistory.length - 1;
        this.updateUndoRedoButtons();
    }

    loadState(state) {
        if (!this.config.modules.human_counter) this.config.modules.human_counter = { strategy: 'humancount', visualize: true, include_classes: ['human'], coordinates: [] };
        this.config.modules.human_counter.coordinates = JSON.parse(JSON.stringify(state.human_counter || []));

        if (this.selectedGroup) {
            this.setPolygonSelectionState(this.selectedGroup, false);
            this.selectedGroup = null;
        }

        this.drawExistingPolygons();
        this.generateConfigForm();
        this.updateToolState();
        this.saveSessionDebounced();
    }

    undo() {
        if (this.actionIndex > 0) {
            this.actionIndex--;
            this.loadState(this.actionHistory[this.actionIndex]);
            this.log('Undo successful.');
            this.updateUndoRedoButtons();
        }
    }

    redo() {
        if (this.actionIndex < this.actionHistory.length - 1) {
            this.actionIndex++;
            this.loadState(this.actionHistory[this.actionIndex]);
            this.log('Redo successful.');
            this.updateUndoRedoButtons();
        }
    }

    updateUndoRedoButtons() {
        const undoBtn = document.getElementById('undo');
        const redoBtn = document.getElementById('redo');
        if (undoBtn) undoBtn.disabled = this.actionIndex <= 0;
        if (redoBtn) redoBtn.disabled = this.actionIndex >= this.actionHistory.length - 1;
    }

    toggleEraser(forceState = null) {
        this.isErasing = forceState !== null ? forceState : !this.isErasing;
        const eraserBtn = document.getElementById('eraserTool');
        if (this.isDrawing) this.cancelCurrentPolygon();

        if (this.isErasing) {
            this.stage.container().style.cursor = 'crosshair';
            if (eraserBtn) eraserBtn.classList.add('tool-active');
            this.log('Eraser tool activated. Click a vertex on the selected polygon to remove it.', 'info');
        } else {
            this.stage.container().style.cursor = 'grab';
            if (eraserBtn) eraserBtn.classList.remove('tool-active');
            this.log('Eraser tool deactivated.', 'info');
        }
    }

    eraseVertex(group, vertexIndex) {
        const areaType = group.getAttr('areaType');
        const areaIndex = group.getAttr('areaIndex');
        
        // Chỉ xử lý các module được phép
        if (!this.allowedModules.includes(areaType)) {
            this.log(`Cannot erase vertex from module "${areaType}" - not allowed.`, 'warning');
            return;
        }

        let coordsList, areaName;
        if (areaType === 'human_counter') {
            coordsList = this.config.modules.human_counter.coordinates;
            areaName = 'Human Counter Area';
        }

        if (coordsList.length <= 3) {
            this.log('Cannot remove vertex. Polygon must have at least 3 points.', 'warning');
            return;
        }

        coordsList.splice(vertexIndex, 1);
        this.log(`Removed vertex from "${areaName}"`);

        this.drawExistingPolygons();
        const newGroup = this.polygons.find(p => p.getAttr('areaType') === areaType && p.getAttr('areaIndex') === areaIndex);

        if (newGroup) {
            this.selectedGroup = newGroup;
            this.setPolygonSelectionState(newGroup, true);
        } else {
            this.selectedGroup = null;
        }

        this.generateConfigForm();
        this.saveState();
        this.saveSessionDebounced();
    }
}

document.addEventListener('DOMContentLoaded', () => {
    window.app = new SmartCityApp();
});