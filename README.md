# AdaptiveEdge: LLM-Powered Multi-Agent Trading System

**Imperial College Algothon 2026 — IMCity Trading Challenge**


## The Idea

Most teams at this Algothon will submit bots built on static, rule-based strategies: fixed spread market making, textbook arbitrage, simple momentum signals. These approaches are well-understood, predictable, and fundamentally limited — they can't adapt when market conditions shift mid-competition.

**AdaptiveEdge takes a fundamentally different approach.** We replace hardcoded trading logic with a live LLM agent (Google Gemini) that reasons about market microstructure in real time. Our bot ingests order book snapshots, trade flow, cross-asset correlations, and its own P&L — then makes strategic trading decisions the way a human quant trader would, but at machine speed.

This isn't just a wrapper around an API call. Our system implements a **multi-agent architecture** inspired by the TradingAgents framework (Xiao et al., UCLA/MIT, 2024), where specialised reasoning modules handle different aspects of the trading problem — market making, arbitrage detection, alpha signal generation, and risk management — and a meta-agent synthesises their outputs into coherent trade execution.

---

## Why This Is Creative

Most traditional LLM-based trading bots use the following strategy: `if spread > threshold: place_order()`. We're asking the LLM to *discover* what the threshold should be, whether the threshold concept even applies right now, and whether there's a better opportunity it should be pursuing instead.

The creative leap is treating the trading bot not as a state machine but as a **reasoning engine**. The bot maintains a running internal model of market state — inventory exposure, volatility regime, observed competitor behaviour — and uses chain-of-thought reasoning to decide its next action. When the market shifts from trending to mean-reverting, our bot doesn't need a regime-detection classifier to tell it so. It *reads the tape* and adapts, just as a human trader would.

We explored many ideas before converging on this approach: recurrent neural networks for price prediction, LSTM-based sequence models, Monte Carlo simulation for path generation, prediction market mechanisms, and internal agent-to-agent trading networks. Each had merit in isolation, but the key insight was that no single hardcoded model could match the flexibility of an LLM that reasons about all of these concepts simultaneously and applies whichever is most relevant to the current market state.

---

## Why This Is Innovative

Our approach draws on cutting-edge research that most teams won't have encountered:

**Multi-Agent LLM Trading (Xiao et al., 2024):** The TradingAgents paper from UCLA and MIT demonstrated that LLM agents with specialised roles — fundamental analyst, sentiment analyst, technical analyst, risk manager — collaborating through structured debate significantly outperform traditional strategies. Their framework achieved Sharpe ratios of 5.6–8.2 across AAPL, GOOGL, and AMZN, dwarfing every baseline. We adapt this multi-agent concept to the real-time, low-latency environment of the CMI exchange.

**Self-Adaptive AI Agents:** Rather than training a model offline and deploying it frozen, our agent continuously refines its strategy based on what's working. It tracks which of its reasoning patterns (market making, arbitrage, momentum) are generating P&L, and dynamically re-weights its approach. This is closer to how modern research on self-adaptive agents in financial markets envisions autonomous trading — not static rules, but live strategic evolution.

**Agentic AI for Finance:** We are leveraging the Gemini API not as a simple text completion engine, but as an autonomous agent with tool access, memory, and structured decision-making. The agent receives market data, maintains conversation history as working memory, and outputs structured trade orders — a genuine agentic loop, not a one-shot prompt.

---

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

Throughout our preparation, we investigated multiple approaches before converging on the LLM agent architecture:

- **Recurrent Neural Networks & LSTMs** for time-series price prediction — powerful for sequence modelling but require training data we don't have access to before the competition starts, and can't adapt to unseen market dynamics.
- **Prediction market mechanisms** where internal agents "bet" on price direction and the consensus drives trading — intellectually elegant but adds latency and complexity without clear advantage over a single reasoning agent.
- **Multi-agent internal trading networks** where specialised bots trade with each other to discover prices — interesting for simulation but doesn't generate real alpha on an external exchange.
- **Monte Carlo and Quantum Monte Carlo simulation** for path generation and option pricing — useful in theory but overkill for a spot-trading competition with no derivatives.
- **Self-adaptive AI agents** informed by recent research on autonomous trading systems — this concept survived and became central to our final architecture.

The common thread: every approach had value as a *concept*, but only the LLM agent could flexibly apply all of these ideas as reasoning frameworks rather than rigid implementations.

---

## Technical Stack

- **Python 3.12** with the CMI Exchange bot framework
- **Google Gemini API** for real-time strategic reasoning
- **Multi-agent prompt architecture** inspired by TradingAgents (arXiv:2412.20138)
- **Structured JSON output** for reliable trade execution
- **Sliding-window market memory** for temporal pattern recognition

---

## Team

Built at the Imperial College Algothon 2026, sponsored by IMC Trading and Man Group.

---

*"The best traders don't follow rules — they reason about markets. We built a bot that does the same."*
