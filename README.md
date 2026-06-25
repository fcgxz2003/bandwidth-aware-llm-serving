# bandwidth-aware-llm-serving

## Offline Algorithm: BACG (Bandwidth-Aware Co-caching Greedy)

`offline/bacg.py` It greedily preheats foundation
models and adapters by repeatedly placing the candidate with the highest
marginal-gain density (delay saved per GB), exploiting submodularity and lazy
evaluation for efficiency.

```mermaid
flowchart TD
    A[Step 3: enumerate candidates = content x cloudlet] --> B[Compute density = gain / size for each candidate]
    B --> C[Push into max-heap]
    C --> D{Heap non-empty?}
    D -->|Yes| E[Pop highest-density candidate]
    E --> F[Recompute true marginal gain]
    F --> G{Still >= runner-up?}
    G -->|No, stale| H[Re-insert with updated density] --> D
    G -->|Yes| I{Budget & storage OK?}
    I -->|No| D
    I -->|Yes| J[Place: write cache + record mu/nu]
    J --> K[Deduct budget + update registry cap]
    K --> L[Refresh D_M / D_W of affected requests]
    L --> D
    D -->|No| M[Return mu, nu]
```
