# Manufacturing Replenishing RL Environment

A small plant builds products to a forecast plan. You are the materials planner: order
components so every production run can run in full, the warehouse never overflows and the
budget holds. Success is binary and checked by a verifier that replays your actions against
the hidden true stock.

## Setup

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/), which fetches Python itself.

```bash
uv run mrpenv serve   # play it yourself at http://127.0.0.1:8000
uv run mrpenv run     # watch the heuristic agent play one episode
uv run mrpenv eval    # benchmark every agent, writes benchmark.png
```

## Environment description

**State** (hidden from the agent)

| Field | Meaning |
|---|---|
| `day` | current day, `0 .. H` |
| `true_stock` | what is physically on the shelf |
| `recorded_stock` | what the ERP believes is on the shelf |
| `purchase_orders` | open orders, with their **actual** arrival day |
| `spend` | committed spend so far |
| `scenario` | the disturbance tables: shrinkage, short deliveries, lateness, rush orders, price paths |

**Observation** (what the agent sees — recorded data only)

| Field | Meaning |
|---|---|
| `day`, `horizon`, `done` | the clock |
| `inventory[]` | `sku`, `recorded_on_hand`, `on_order`, `unit_price_today`, `unit_price_change` |
| `open_purchase_orders[]` | `po_id`, `sku`, `qty`, `placed_day`, `promised_arrival` (never the actual one) |
| `production_log[]` | `day`, `product`, `planned` (forecast), `required` (realised), `completed` |
| `spend`, `budget_remaining` | money committed and left |
| `capacity_m3`, `recorded_utilisation` | warehouse size and how full the records say it is |
| `steps_used`, `step_limit`, `last_events` | turn budget and recent events |

Static, fetched once: `horizon`, `capacity_m3`, `budget_eur`, `suppliers` (lead time, lot size,
MOQ, unit cost, unit volume), `bom`, `mps`, `requirements`. The difficulty tier is never shown.

**Actions** — one tool call is one step; reads cost a turn but no money.

| Tool | Args | Effect |
|---|---|---|
| `view_inventory` | – | recorded stock, on-order, open POs, today's quotes |
| `view_plan` | `from_day?`, `to_day?` | forecast plan and derived component requirements |
| `view_master_data` | – | BOM, supplier table, capacity, budget |
| `view_prices` | – | today's quotes plus the history so far |
| `create_purchase_order` | `sku`, `qty` | order placed today, arrives after the lead time |
| `advance_day` | – | receipts, then production consumes its components |

**Reward** — `0` on every step, then at day `H` a single binary score, `1` only if all of:

| Check | Condition |
|---|---|
| integrity | the action ledger's HMAC chain is intact and the reported end state matches the replay |
| complete | the episode covered the whole horizon |
| valid | no tool call was rejected by the ERP |
| service | true stock covered the realised requirement on **every** production day |
| capacity | true end-of-day volume never exceeded the warehouse |
| budget | total spend stayed within budget |

**Initial state distribution** — every scenario is drawn from its seed and then rejection-sampled
against a clairvoyant planner, so warehouse capacity and budget are derived from that plan's own
trajectory and every emitted instance is provably solvable.

**Tiering** — three tiers (`easy`, `medium`, `hard`) raise the number of products, the horizon and
the disturbances together, giving a curriculum from a clean single line to a noisy ERP with a
volatile market.

## Benchmark

![Success rate by tier](benchmark.png)

## Design notes

The environment is built like a robotics sim, not like a business simulator. The plant is the
"physics", the ERP is a sensor, and the interesting difficulty is that the two disagree.

| What this environment does | Where the idea comes from |
|---|---|
| The simulator keeps hidden `true_stock`; the agent only ever sees `recorded_stock`, which drifts from it through unbooked scrap and short deliveries. The drift is **biased one way** — the ERP always overstates the shelf — so trusting it causes stockouts. | The privileged-state / state-estimate split. A robot acts on a noisy estimate, not on ground truth, and sensor error is rarely zero-mean. |
| Randomising *data pathologies* — record error, supplier lateness, rush orders, price volatility — rather than friction and mass. | Domain and dynamics randomisation (Tobin et al. 2017; Peng et al. 2018), moved from physics to enterprise data. |
| Every disturbance is drawn once from the seed into a fixed table before the episode starts, so it cannot react to the agent, and two rollouts on one seed face exactly the same world. | Common random numbers for fair policy comparison and deterministic replay (Ng & Jordan's PEGASUS, 2000); disturbance rejection in control. |
| A clairvoyant planner that reads the hidden tables is used to *certify* instances by rejection sampling, so a zero always means the policy failed, never that the instance was impossible. | Privileged teachers (Chen et al. 2020; Lee et al. 2020) — used here for feasibility rather than for distillation. |
| Reward is computed on the replayed **true** trajectory, never on the records. Scoring the records instead is a documented exploit: a correct planner passes while the factory starves. | Reward from simulator ground truth, not from the observation, so the policy cannot learn to satisfy its own sensor. |
| Binary, conjunctive success — served *and* within capacity *and* within budget. No partial credit. | Task-completion scoring in manipulation benchmarks. Weighted partial credit creates exchange rates between failures, and an optimiser finds them. |
| A separate verifier replays the HMAC-chained action log with a second, independent implementation of the dynamics and recomputes every number it is given. | Differential testing of safety-critical controllers, plus offline evaluation from logged rollouts. |
| The buffers a good policy needs cannot be triggered by evidence: lateness is observable only after the late delivery, so a prior has to carry the early days and observation only refines it. | Latent environment parameters estimated online from a history of observations (RMA, Kumar et al. 2021). Our first agent waited for evidence and scored no better than the naive one. |
| Three tiers raise all the disturbances together, and an episode is a dict of Python objects, so hundreds run per container. | Curriculum over randomisation ranges (OpenAI's automatic domain randomisation, 2019) and massively parallel simulation (Isaac Gym, 2021). |
