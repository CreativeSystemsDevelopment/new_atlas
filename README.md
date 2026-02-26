# ATLAS — Industrial Schematic Digital Twin



Component annotation tool for identifying electrical schematic components and their labels.

## Quick Start

```bash
pip install -r requirements.txt
python tools/annotator/annotator.py "path/to/schematic.pdf"
```

## Workflows

### 1. Component Annotation (Default)
- **Left-drag**: Draw box around component (snaps to vector shapes)
- **Left-click**: Click text label to associate with component
- **Output**: Green box + Blue label box

### 2. Circuit Continuation
- **Ctrl+D** or **⊙ Continuation** button: Enter mode
- **Left-drag**: Draw box around continuation symbol (Amber)
- **Click**: Page number text
- **Click**: Rung number text
- **Output**: Amber symbol + Page/Rung refs

### 3. Wire Tracing
- **W** or **⚡ Wire** button: Enter mode
- **Click**: Wire label text (e.g. "24V", "0V")
- **Output**: Purple label box (saved for vector tracing)

### 4. Touch Gestures (Tablet/Stylus)
- **Pinch**: Zoom in/out
- **Flick Left/Right**: Previous / Next page
- **Side Button + Drag**: Pan view
- **Eraser Tap**: Redo
- **Eraser Double-Tap**: Undo

## Controls
| Key | Action |
|---|---|
| `Scroll` | Zoom in/out |
| `Right-Drag` | Pan view |
| `←` / `→` | Previous / Next page |
| `F` | Fit to view |
| `Ctrl+Z` | Undo last action |
| `Ctrl+Y` | Redo |
| `Del` | Delete selected annotation |

## Project Structure

```
new_atlas/
│   └── annotator/
│       ├── annotator.py   # Native Desktop App (PySide6)
│       ├── preprocess.py  # PDF Processing (1200 DPI + metadata)
│       ├── annotations/   # JSON output per page
│       └── data/          # Processed images and metadata
├── resources/             # Schematic PDFs and symbol libraries
└── requirements.txt
```

## Full repo index

For a practical file-by-file map and feature index, see [`PROJECT_INDEX.md`](PROJECT_INDEX.md).
