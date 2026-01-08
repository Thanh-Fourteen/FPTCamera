# FPTCamera
A comprehensive computer vision system for special camera monitoring and analysis.


## Structure repository
```
FPTCamera/
│
├── README.md                  # General project introduction
├── requirements.txt           # Python libraries required
├── docs/                      # Documentation
├── logs/                      # Saving logs
├── tests/                     # Test unit/integration
└── src/
    ├── __init__.py
    ├── pipeline/              # Pipeline 
    │   ├── __init__.py
    ├── common/                # Shared
    │   ├── logger.py          # Config logging
    │   └── visualizer.py      # Visualize frame
    │   └── loader.py          # Load config
    └── components/            # Core components
        ├── detection/         # Yolo Triton
        └── tracking/          # BoT-SORT / ByteTrack
```

