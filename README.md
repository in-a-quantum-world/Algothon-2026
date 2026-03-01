# AdaptiveEdge: LLM-Powered Multi-Agent Trading System

**Imperial College Algothon 2026 — IMCity Trading Challenge**


## The Idea

Most teams at this Algothon will submit bots built on static, rule-based strategies: fixed spread market making, textbook arbitrage, simple momentum signals. These approaches are well-understood, predictable, and fundamentally limited — they can't adapt when market conditions shift mid-competition.

**We take a fundamentally different approach.** We replace hardcoded trading logic with a live LLM agent (Google Gemini) that reasons about market microstructure in real time. Our bot ingests order book snapshots, trade flow, cross-asset correlations, and its own P&L — then makes strategic trading decisions the way a human quant trader would, but at machine speed.

This isn't just a wrapper around an API call. Our system implements a **multi-agent architecture** inspired by the TradingAgents framework (Xiao et al., UCLA/MIT, 2024), where specialised reasoning modules handle different aspects of the trading problem — market making, arbitrage detection, alpha signal generation, and risk management — and a meta-agent synthesises their outputs into coherent trade execution.

---

## Why This Is Creative

Most traditional LLM-based trading bots use the following strategy: `if spread > threshold: place_order()`. We're asking the LLM to *discover* what the threshold should be, whether the threshold concept even applies right now, and whether there's a better opportunity it should be pursuing instead.

The creative leap is treating the trading bot not as a state machine but as a **reasoning engine**. The bot maintains a running internal model of market state — inventory exposure, volatility regime, observed competitor behaviour — and uses chain-of-thought reasoning to decide its next action. When the market shifts from trending to mean-reverting, our bot doesn't need a regime-detection classifier to tell it so. It *reads the tape* and adapts, just as a human trader would.

We explored many ideas before converging on this approach: recurrent neural networks for price prediction, LSTM-based sequence models, Monte Carlo simulation for path generation, prediction market mechanisms, and internal agent-to-agent trading networks. Each had merit in isolation, but the key insight was that no single hardcoded model could match the flexibility of an LLM that reasons about all of these concepts simultaneously and applies whichever is most relevant to the current market state.

---


## Research Foundations

- **TradingAgents** (Xiao et al., UCLA/MIT, 2024, arXiv:2412.20138): Multi-agent LLM framework with specialised analyst, researcher, trader, and risk manager roles. Achieved Sharpe ratios of 5.6–8.2 across AAPL, GOOGL, AMZN over June–November 2024, outperforming buy-and-hold, MACD, RSI, and SMA baselines by 6–25 percentage points in cumulative returns. We adopt the multi-role reasoning concept but collapse it into a single prompt with role-specific instruction sections, trading inter-agent communication overhead for lower latency.
- **Emergent AI collusion in markets** (Wharton School, 2024): demonstrated that independent RL trading agents spontaneously develop coordinated pricing without explicit communication. Motivated our design choice to use an LLM that can reason about other agents' likely behaviour (e.g., "these quote patterns look like another market-making bot — avoid getting picked off") rather than treating the order book as exogenous.
- **Avellaneda-Stoikov market making** (2008): the inventory skewing and spread calculation logic follows this framework, but with LLM-determined parameters rather than analytically derived constants, allowing adaptation to the specific microstructure of the CMI exchange.

---

## Technical Stack

- Python 3.12, CMI Exchange `BaseBot` framework (SSE stream, REST order API)
- Google Gemini 2.0 Flash API (`google-genai` SDK) with structured JSON output mode
- NumPy for rolling statistics: Pearson correlation, z-score, EWMA, rolling σ
- Threading: SSE stream on daemon thread, Gemini calls rate-limited on main loop


## Why This Has Promise

The evidence for LLM-based trading is compelling and recent:

- The TradingAgents framework achieved **26.6% cumulative returns on AAPL** over a 6-month period where buy-and-hold lost 5.2%, with a Sharpe ratio of 8.21 — an order of magnitude above traditional strategies.
- LLMs can synthesise heterogeneous information (order book microstructure, trade flow patterns, cross-asset signals) in a way that no single classical model can. A mean-reversion model can't reason about order book imbalance. An arbitrage detector can't assess inventory risk. An LLM can do all of these simultaneously.
- The competition format — real-time trading on a simulated exchange with correlated assets — is *precisely* the kind of multi-dimensional problem where rigid rules underperform flexible reasoning. Fixed-spread market makers get picked off by informed flow. Static arbitrage strategies get front-run. An adaptive agent sidesteps these failure modes.
- Our self-refinement loop means the bot gets *better* as the competition progresses. Early trades are exploratory; later trades are informed by accumulated market intelligence. Teams running static strategies have no equivalent mechanism.

---

## How We're Different From Other Teams

| Dimension | Typical Team | AdaptiveEdge |
|---|---|---|
| **Strategy source** | Hardcoded rules from textbooks | LLM-generated reasoning, adapted live |
| **Adaptability** | Fixed parameters | Continuously evolving strategy |
| **Market making** | Static spread, fixed skew | Dynamic spread adjusted by LLM based on flow toxicity and inventory |
| **Arbitrage** | Simple threshold crossing | Multi-factor reasoning about *why* prices diverged and whether the opportunity is real |
| **Risk management** | Position limits and stop-losses | Holistic risk assessment considering correlation, volatility regime, and portfolio exposure |
| **When conditions change** | Breaks or underperforms | Reasons about the change and adapts |
| **Research backing** | Standard quant textbooks | State-of-the-art multi-agent LLM trading research (2024) |

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              CMI Exchange (SSE Stream)           │
│         Order Books · Trades · Positions         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│            Market State Aggregator               │
│  Order book snapshots · Trade history · P&L      │
│  Cross-asset correlations · Inventory tracker    │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│         Gemini LLM Agent (Multi-Role)            │
│                                                  │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │  Market     │ │ Arbitrage  │ │   Alpha      │ │
│  │  Maker      │ │ Detector   │ │   Signal     │ │
│  └────────────┘ └────────────┘ └──────────────┘ │
│  ┌────────────┐ ┌────────────────────────────┐   │
│  │   Risk     │ │    Meta-Strategy Agent     │   │
│  │  Manager   │ │  (synthesise + decide)     │   │
│  └────────────┘ └────────────────────────────┘   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│            Order Execution Layer                 │
│     Structured JSON → CMI API → Exchange         │
└─────────────────────────────────────────────────┘
```

---

## Ideas We Explored


## Ideas Explored During Development

We investigated several approaches before converging on the LLM agent architecture:

**Recurrent Neural Networks / LSTMs** for price prediction: we prototyped an LSTM trained on rolling windows of mid-price returns to predict next-tick direction. The problem: without pre-competition training data from the CMI exchange's specific price dynamics, the model would be fitting to a distribution it hasn't seen. LSTMs are powerful sequence models but need representative training data, which isn't available until the competition starts. The LLM approach sidesteps this because it reasons about market structure rather than learning statistical patterns from historical data.

**Prediction market mechanisms**: we considered an internal prediction market where multiple sub-agents "bet" on the next direction and the consensus drove trade sizing. Connects to research on wisdom-of-crowds in forecasting, but adds ~500ms of internal computation per decision. The LLM already implicitly aggregates multiple analytical perspectives in a single call, achieving the same effect with one API round-trip instead of multiple.

**Multi-agent internal trading networks**: inspired by research on emergent behaviour in agent-based market simulations (Wharton School, 2024), we explored having specialised agents trade with each other to discover internal prices. However, internal trades between your own agents are zero-sum — they move capital around within your system without generating alpha on the external exchange. The valuable insight (that different strategy viewpoints should inform each other) was preserved by making the LLM reason across all strategies simultaneously in a single prompt.

**Monte Carlo path simulation**: considered generating price path distributions to estimate fair value and optimal hedge ratios. Useful for derivatives pricing but the CMI exchange trades spot products, and 10,000 path simulations at ~100ms competes with the LLM call for latency budget. We chose to allocate that latency to the LLM's broader reasoning capability instead.

**Self-adaptive AI agents**: research on autonomous trading agents that modify their own behaviour based on observed market regime changes (switching between trend-following and mean-reversion parametrically) was directly incorporated into the self-adaptation mechanism described above. The LLM serves as the adaptation controller, adjusting strategy parameters based on P&L feedback rather than requiring a separate regime classifier.



## Team

Built at the Imperial College Algothon 2026, sponsored by IMC Trading and Man Group.

---

*"The best traders don't follow rules — they reason about markets. We built a bot that does the same."*
