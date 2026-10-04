# Dynamic Pricing for Ride-Sharing — Reinforcement Learning & Genetic Algorithm

**Module:** H9AIDM (AI-Driven Decision Making) — CA2  
**Programme:** MSc Artificial Intelligence, National College of Ireland (NCI)  
**Author:** Prasanth Oddepally — x25205595s@student.ncirl.ie

## 📌 Overview

This project models a **dynamic surge-pricing problem** for ride-sharing services (Lyft) and solves it using two complementary AI approaches:

1. **Reinforcement Learning — Tabular Q-learning:** Learns a surge-pricing policy that maximises expected revenue while balancing rider goodwill.
2. **Evolutionary Optimisation — Genetic Algorithm (GA):** Evolves a pricing policy directly through selection, crossover and mutation over an exact, factorised fitness function.

Both learned policies are evaluated against fixed-price, always-max, and rule-based heuristics, plus an oracle reference, and are stress-tested for robustness under demand-elasticity mis-specification and held-out generalisation.

## 📊 Dataset

The project uses the **Uber & Lyft Dataset — Boston, MA** from Kaggle:

🔗 **https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma**

Key details:

- **Provider:** Kaggle user `brllrb`
- **Content:** Ride-level records for Uber and Lyft in the Boston, MA area
- **Key fields used:** `datetime`, `hour`, `day`, `month`, `source`, `destination`, `cab_type`, `name`, `price`, `distance`, `surge_multiplier`, `short_summary`, `temperature`, `precipIntensity`, `windSpeed`
- **Filtering in this notebook:** only **Lyft** rides are used, with valid positive prices, positive distances, and surge multipliers in `{1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}`

### Setting the data path

Place the raw Kaggle CSV (`rideshare_kaggle.csv`) somewhere accessible and update `DATA_PATH` in the notebook:

```python
DATA_PATH = r"C:\Users\msi\OneDrive\Desktop\H9AIDM\Data\rideshare_kaggle.csv"
```

> **Fallback:** If the raw CSV is not found, the notebook will try `clean_rides.csv` next to the notebook or in `outputs/`, and finally fall back to the pre-existing `outputs/demand_table.csv`.

## 📁 Repos Structure

```
H9AIDM_CA2_Oddepally_x25205595/
├── AIDM CA2_main.ipynb          # Main analysis notebook (all 6 sections)
├── AIDM CA2_presentation.pptx   # Project presentation
├── Video link.md                # Video walkthrough link
├── README.md                    # This file
├── Data/
│   └── rideshare_kaggle.csv     # Raw Kaggle dataset (optional, local copy)
├── Figures/                     # Report-quality figures
└── outputs/                     # Generated outputs (CSVs, learned policies, plots)
    ├── demand_table.csv         # Demand-context aggregation
    ├── base_price_model.csv     # Per-product base-price linear model
    ├── q_policy.csv             # Learned Q-learning surge policy
    ├── ga_policy.csv            # GA-evolved surge policy
    ├── q_table.npy              # Learned Q-table
    ├── results_summary.csv      # Policy comparison results
    ├── sensitivity_summary.csv  # Elasticity sensitivity results
    └── *.png                    # EDA + convergence + comparison figures
```
## Getting Started

### Requirements
- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
Install dependencies:
```bash
pip install numpy pandas matplotlib
```
### Running the notebook
1. Download the Kaggle dataset and place the CSV at the path in `DATA_PATH` (or add `clean_rides.csv` to the project root).
2. Open `AIDM CA2_main.ipynb` in Jupyter Notebook / VS Code / Google Colab.
3. Run the cells **top to bottom** — all outputs are written to the `outputs/` directory.

## 📓 Notebook Sections

| Section | Description |
|---------|-------------|
| **1 · Data preparation** | Loads, cleans and feature-engineers the raw ride data: time-of-day buckets, weather simplification, base-price estimation, demand-context aggregation and EDA plots. |
| **2 · Pricing environment (MDP)** | Builds a Gym-style Markov Decision Process with state = (demand level, weather, product, goodwill bucket), exponential demand response, churn/recovery dynamics, and a discounted return metric plus an oracle baseline. |
| **3 · Q-learning agent (RL)** | Trains a tabular Q-learning agent (60k episodes) with ε-greedy exploration, decaying learning rate, TD-error and policy-churn diagnostics, and a learned-policy heatmap. |
| **4 · Genetic algorithm** | Evolves a policy genome (120 individuals, 150 generations) with tournament selection, uniform crossover and mutation over an exact, vectorised fitness function. |
| **5 · Evaluation & robustness** | Noisy rollout comparison with 95% CIs, paired per-episode significance tests, ±20% elasticity sensitivity analysis, and 70/30 held-out generalisation check. |
| **6 · Summary of results** | Prints the final policy comparison table and lists all generated outputs. |

---

## 🔬 Methodology Summary

### State representation

```
state = (demand_level ∈ {low, medium, high},
         weather      ∈ {clear, cloudy, rain},
         product      ∈ {Shared, Lyft, Lyft XL, Lux, Lux Black, Lux Black XL},
         goodwill     ∈ {5 buckets})
```
### Assumed demand model
Demand responds to a surge multiplier `a` via exponential decay:
```
Q(a) = Q_base × goodwill × exp(−λ · (a − 1.0))
```
where `λ` depends on demand level, weather and product tier, and goodwill evolves with churn (`5%` per surge unit) and recovery (`22%` per step).

### Reward

```
reward = fare × q − c_dissat × base_price × (a − 1.0) × q
```
i.e. revenue minus a customer-dissatisfaction penalty for excessive surge.

## ✅ Key Results

The learned policies aim to beat the **fixed 1.0× surge baseline**; the notebook reports:

- Q-learning and GA both achieve substantial improvements over fixed pricing in expected discounted return
- Statistical significance is assessed via paired per-episode tests (95% CI, % episodes improved)
- Robustness results show performance under mis-specified elasticity (±20%)
- Held-out generalisation quantifies the train/test gap on unseen demand contexts

All numbers, figures and policy files are regenerated automatically in `outputs/`.

## 📎 Links

- **Kaggle Dataset:** https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma
- **Video Walkthrough:** See `Video link.md`

