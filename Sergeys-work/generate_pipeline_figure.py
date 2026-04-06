"""Generate pipeline flow diagrams as Mermaid markdown files.

Produces .md files with Mermaid diagrams that can be rendered in GitHub,
VS Code preview, or any Mermaid-compatible viewer.

Output: synthetic_slides/00_pipeline_flow.md, real_world_slides/00_pipeline_flow.md
"""

import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SYNTHETIC_DIAGRAM = """```mermaid
flowchart TD
    subgraph SOURCE["Input Source"]
        S["Synthetic Scene Model<br/><i>Generated trajectories<br/>velocity, object size, background</i>"]
    end

    subgraph DVS_PATH["DVS Path"]
        direction TB
        D1["DVS Sensor Simulation<br/><i>Log-intensity frame diff → events<br/>Per-pixel, asynchronous</i>"]
        D2["DVS Detection<br/><i>Event accumulation →<br/>connected components / MOG2</i>"]
        D3["DVS Power Model<br/><i>P_static + event_rate × E_per_event</i>"]
    end

    subgraph CIS_PATH["CIS Path"]
        direction TB
        C1["CIS Sensor Simulation<br/><i>Frame subsampling + noise<br/>Resolution, ADC, motion blur</i>"]
        C2["CIS Detection<br/><i>MOG2 / KNN /<br/>Optical Flow / Public Dets</i>"]
        C3["CIS Power Model<br/><i>P_idle + FPS_frac × (P_max − P_idle)</i>"]
    end

    subgraph SHARED["Shared Pipeline"]
        T["Tracker<br/><b>SORT</b> (Kalman + Hungarian)<br/><b>IoU</b> / <b>Centroid</b>"]
        E["Evaluation<br/><b>MOTA</b> / <b>IDF1</b> / <b>ID Switches</b><br/><i>Hungarian matching vs ground truth</i>"]
    end

    subgraph OUTPUT["Output"]
        O["MOTA vs Power per Sensor<br/>→ Crossover Analysis<br/>→ Design Rule"]
    end

    S --> D1
    S --> C1
    D1 -->|"latency, θ, refractory cap"| D2
    C1 -->|"FPS, resolution, ADC bits"| C2
    D1 --> D3
    C1 --> C3
    D2 --> T
    C2 --> T
    T --> E
    E --> O
    D3 --> O
    C3 --> O

    classDef source fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef dvs fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef cis fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    classDef shared fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C
    classDef output fill:#FBE9E7,stroke:#BF360C,stroke-width:2px,color:#BF360C

    class S source
    class D1,D2,D3 dvs
    class C1,C2,C3 cis
    class T,E shared
    class O output
```
"""

REALWORLD_DIAGRAM = """```mermaid
flowchart TD
    subgraph SOURCE["Input Source"]
        S["MOT17 Video Dataset<br/><i>Real frames + ground truth<br/>1920×1080, 30 fps, MOT17-04-SDP</i>"]
    end

    subgraph DVS_PATH["DVS Path"]
        direction TB
        D1["DVS Sensor Simulation<br/><i>v2e emulator / frame-diff fallback<br/>Per-pixel, asynchronous events</i>"]
        D2["DVS Detection<br/><i>Event accumulation →<br/>connected components / MOG2</i>"]
        D3["DVS Power Model<br/><i>P_static + event_rate × E_per_event</i>"]
    end

    subgraph CIS_PATH["CIS Path"]
        direction TB
        C1["CIS Sensor Simulation<br/><i>Frame subsampling + noise<br/>Resolution, ADC, motion blur</i>"]
        C2["CIS Detection<br/><i>MOG2 / KNN /<br/>Optical Flow / Public Dets</i>"]
        C3["CIS Power Model<br/><i>P_idle + FPS_frac × (P_max − P_idle)</i>"]
    end

    subgraph SHARED["Shared Pipeline"]
        T["Tracker<br/><b>SORT</b> (Kalman + Hungarian)<br/><b>IoU</b> / <b>Centroid</b>"]
        E["Evaluation<br/><b>MOTA</b> / <b>IDF1</b> / <b>ID Switches</b><br/><i>Hungarian matching vs ground truth</i>"]
    end

    subgraph OUTPUT["Output"]
        O["MOTA vs Power per Sensor<br/>→ Crossover Analysis<br/>→ Design Rule"]
    end

    S --> D1
    S --> C1
    D1 -->|"latency, θ, refractory cap"| D2
    C1 -->|"FPS, resolution, ADC bits"| C2
    D1 --> D3
    C1 --> C3
    D2 --> T
    C2 --> T
    T --> E
    E --> O
    D3 --> O
    C3 --> O

    classDef source fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef dvs fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef cis fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    classDef shared fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#4A148C
    classDef output fill:#FBE9E7,stroke:#BF360C,stroke-width:2px,color:#BF360C

    class S source
    class D1,D2,D3 dvs
    class C1,C2,C3 cis
    class T,E shared
    class O output
```
"""


def main():
    for folder, diagram, title in [
        ("synthetic_slides", SYNTHETIC_DIAGRAM, "Synthetic Pipeline Flow"),
        ("real_world_slides", REALWORLD_DIAGRAM, "Real-World Pipeline Flow"),
    ]:
        out_dir = os.path.join(THIS_DIR, folder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "00_pipeline_flow.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(diagram)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
