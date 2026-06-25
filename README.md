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

## Online Algorithm: DEWMA (Dual-EWMA continuous preheating)

`online/dewma.py` continuously preheats foundation models and adapters slot by
slot. For each time slot it serves the current requests, updates a dual-EWMA
demand estimator (a short-term R-EWMA blended with a daily-periodic D-EWMA),
predicts the next slot's demand, and invokes the offline BACG greedy under the
current slot's idle-bandwidth budget to decide what to preheat.

```mermaid
flowchart TD
    A[Slot t: incoming requests] --> B[Compute current-slot metrics: pull delay, hit rate, BTS]
    B --> C[Serve & cache served content via LRU]
    C --> D[Count demands per cloudlet x model x service]
    D --> E[Update dual-EWMA: R-EWMA short-term + D-EWMA daily]
    E --> F[Estimate next-slot demand = blend by theta]
    F --> G[Build predicted request set: fractional -> integer rounding]
    G --> H{Any predicted requests?}
    H -->|No| I[idle_bw = 0 for this slot] --> N{More slots?}
    H -->|Yes| J[Invoke offline BACG greedy under slot bandwidth budget]
    J --> K[Receive preheat decisions mu, nu]
    K --> L[Refresh LRU recency of preheated content]
    L --> M[Record idle bandwidth consumed this slot]
    M --> N
    N -->|Yes| A
    N -->|No| O[Return pull_times, hit_rates, bts_volumes, idle_bw]
```
