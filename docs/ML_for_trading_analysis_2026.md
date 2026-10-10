# ML for Short-Horizon Directional Prediction in BTC Perps and Equities: State of the Field, Late 2026

**Bottom line: nothing released since mid-2025 changes the core fact your walk-forward tests keep showing you. Minute-level direction in liquid assets carries only a tiny edge, a few AUC points at best, and it is usually smaller than trading costs. Newer foundation models (Chronos-2, TimesFM-2.5/3, TiRex-2, Toto 2.0) are much better *general* forecasters. But the only independent finance evaluations find them no better than, or worse than, gradient-boosted trees and random-walk baselines on returns. Your next unit of effort should go into targets, sampling, cost-aware labelling and multiple-testing discipline, with LightGBM as the workhorse, not into another architecture.**

## TL;DR

- **Foundation models will very probably not rescue the TFT setup.** Chronos-2, TimesFM-2.5/3, TiRex-2 and Toto 2.0 lead general benchmarks by a wide margin over seasonal-naive and statistical baselines. On financial returns, though, independent studies find off-the-shelf models underperform CatBoost/LightGBM (zero-shot Chronos-large R²_OOS −1.37% vs CatBoost −0.03% on U.S. daily returns). Fine-tuning barely helps. Only finance-native pretraining closes the gap.
- **Gradient-boosted trees on engineered features should be the primary model, and a trivial model should be the bar to beat.** G-Research reports that all top three finishers of its Crypto Forecasting competition, including winner Meme Lord Capital, used LightGBM, and it credits their feature engineering most. A 2026 matched study found a 3-parameter sign-reversal logit gets a mean AUC of 0.531 on 15-minute crypto bars (52.3% accuracy on BTC). Its gross edge is about 1.3 bp per trade against a 5 bp round-trip cost. Any model you build must beat that baseline *after costs*.
- **Methodology beats architecture, but not every López de Prado tool is proven.** Purged/embargoed validation, deflated Sharpe and trial accounting are well-founded and essential for a cronjob that tries many feature sets. Dollar bars, triple-barrier labels and meta-labelling are sensible ways to reframe the problem. Their "large improvement" evidence, however, comes mostly from vendors or is unreplicated. Treat them as pre-registered hypotheses, not known wins.

## Key Findings

1. **The foundation-model field moved fast from mid-2025 to 2026.** Chronos-2 (Oct 2025), TimesFM-2.5 (Sep 2025) and TimesFM-3 (Aug 2026), TiRex (NeurIPS 2025) and TiRex-2 (Jul 2026), Toto 2.0 (Apr 2026), Moirai-2, FlowState, TabPFN-TS-3 and others added native covariates, multivariate support and quantile outputs. On fev-bench, Chronos-2 has a MASE skill score of 35.5% over seasonal naive. LightGBM scores 20.48 and TFT 18.37.
2. **Those benchmarks contain almost no near-efficient price series.** They also have known contamination issues: GIFT-Eval pretraining overlap, and self-reported leakage on fev-bench (10% for TimesFM-2.5). In short, benchmark wins say little about BTC direction.
3. **The finance-specific evidence is negative to weakly positive.** Rahimikia, Ni & Wang (Nov 2025) find TSFMs weak zero-shot and after fine-tuning, and competitive only when pretrained from scratch on financial data. Noguer i Alonso & Pereira Franklin (Jun 2026) find TSFMs beat locally trained deep nets, but rarely beat a random walk. Only 2 of their model–asset pairs pass a one-sided Diebold–Mariano test at p≈0.04, uncorrected.
4. **Order-flow imbalance is real but mostly contemporaneous.** The OFI literature (Cont–Kukanov–Stoikov 2014; Cont–Cucuringu–Zhang 2023) establishes strong *contemporaneous* explanatory power and modest *predictive* power. Deep LOB models (DeepLOB and successors) degrade sharply out of sample (LOBCAST, 2024).
5. **The edge is smaller than the cost.** The best-documented recent crypto short-horizon signal is gross 1.3 bp against a 5 bp round-trip cost. Directional accuracy in the 51–53% range is what an honest result looks like. Headline 70–90% accuracies almost always come from leakage, benchmark artefacts (FI-2010), or ignoring costs and label overlap.

## Details

### 1. Time-series foundation models and recent architectures

**Pretrained foundation models (zero-shot capable)**

| Model | Released | By | Weights / licence | Size | Probabilistic | Covariates / multivariate | Notes |
|---|---|---|---|---|---|---|---|
| **Chronos-2** | 20 Oct 2025 | Amazon | Open (Hugging Face) | 120M, encoder-only | Multi-step quantiles | Native past-only and known-future covariates; multivariate via group attention | #1 on fev-bench (MASE skill 35.5, win rate 80.9%). Fine-tuning supported; upgrade to chronos-forecasting ≥2.1.0 (past-covariate fine-tuning bug fixed) |
| Chronos / Chronos-Bolt | 2024 | Amazon | Open | tiny–large | Quantiles | Univariate only; covariates via external regressor | Amazon Science reported (Oct 2025) that Chronos and Chronos-Bolt had been "collectively downloaded over 600 million times from Hugging Face" |
| **TimesFM-2.5** | 15 Sep 2025 | Google Research | Apache-2.0 | 200M (down from 500M) | Optional 30M continuous quantile head | Covariates via XReg (restored 29 Oct 2025); univariate core | 16k context. LoRA/PEFT fine-tuning example added Apr 2026 |
| **TimesFM-3** | 31 Aug 2026 | Google Research | **Non-commercial, non-production weights** (code Apache-2.0) | 330M; >1T training points | 9 quantiles | Native multivariate, past and future covariates | Google claims top rank on GIFT-Eval, fev-bench and TIME, shown only as rank charts. No finance evaluation yet |
| **TiRex** | May 2025 (NeurIPS 2025) | NX-AI (Hochreiter) | Open | 35M, xLSTM (recurrent) | Quantiles | Univariate | Very fast; strong short and long horizons |
| **TiRex-2** | Jul 2026 | NX-AI | Open (a "Pro" tier also exists) | small | Quantiles | Multivariate, past and future-known covariates, streaming | #2 on fev-bench (skill 33.74) |
| Toto 1.0 | May 2025 | Datadog | Apache-2.0 | 151M (per Datadog's arXiv paper) | Yes | Exogenous covariates and fine-tuning added Feb 2026 | Observability-focused |
| **Toto 2.0** | Apr 2026 (per DataDog's GitHub) | Datadog | Apache-2.0 | Five sizes: 4M, 22M, 313M, 1B, 2.5B | Quantiles | Multivariate; **no fine-tuning or exogenous support yet in 2.0** | First clean scaling results for TSFMs. Trained only on observability and synthetic data |
| Moirai-2.0 | 2025 | Salesforce | Open | small | Quantile (pinball) | Multivariate | 28% self-reported leakage on fev-bench |
| FlowState | late 2025 | IBM | Open | 9.1M, state-space | Yes | — | #2 zero-shot on GIFT-Eval at its release despite tiny size |
| TabPFN-TS-3 | 2026 | Prior Labs | Open | — | Yes | Known covariates via tabular framing | Slow: median ~235 s per task on fev-bench |
| Lag-Llama | 2024 | ServiceNow/academic | Open | small | Yes | Univariate | Largely superseded |
| TimeGPT | 2023– | Nixtla | **Closed API** | — | Yes | Exogenous supported | Cannot be self-hosted or inspected |
| Sundial, YingLong, TTM-R3, Timer-S1, Super-Linear | 2025–26 | Tsinghua, Alibaba, IBM, others | Mostly open | 2.5M–300M | Mostly | Varies | Mid-table on GIFT-Eval |
| FinCast | Aug 2025 | academic | Open | — | — | — | A finance-specific TSFM evaluated on crypto, FX, stock and futures series from minute to weekly frequency. Promising in concept; no independent replication found |

**Compute.** All open TSFMs above except the 1B/2.5B Toto variants run zero-shot inference comfortably on a 16 GB RTX 5080 laptop GPU. LoRA fine-tuning of Chronos-2 (120M) or TimesFM-2.5 (200M) is feasible there too. Make sure your PyTorch/CUDA build supports Blackwell GPUs.

**Licensing trap.** If this ever becomes a production or commercial project, TimesFM-3's pretrained weights are restricted to non-commercial, non-production use. Chronos-2, TimesFM-2.5, TiRex and Toto are the safe choices.

**Supervised architectures (train-from-scratch)**

| Family | Key models (venue/year) | What they showed | Current standing |
|---|---|---|---|
| Linear | DLinear/NLinear (AAAI 2023), RLinear, Super-Linear (pretrained mixture of linear experts, 2025–26) | A one-layer linear map matched or beat 2021–22 Transformers on LTSF | Still an obligatory baseline |
| Patch Transformers | PatchTST (ICLR 2023) | Patching plus channel independence rescued Transformers on LTSF | Solid baseline. fev-bench skill 17.52, below LightGBM |
| Inverted / variate-token | iTransformer (ICLR 2024) | Attention across variates | Won 1 of 5 equity tasks in the 2026 finance study |
| MLP mixers | TSMixer (Google, 2023), TimeMixer (ICLR 2024) | Competitive with Transformers at lower cost | Mid-pack |
| CNN / frequency | TimesNet (ICLR 2023) | 2D periodicity modelling | Mid-pack; heavy |
| Recurrent | xLSTM, xLSTMTime (2024), TiRex (pretrained) | Recurrent state-tracking is competitive again | TiRex is the strongest evidence |
| State-space | Mamba / S-Mamba (2024), FlowState (IBM, 2025) | Linear-time long context | FlowState is strong for its size |
| KAN | KAN-based forecasters (2024–25) | Interpretability claims | No robust evidence of superiority; treat as experimental |
| TFT | Lim et al. (2021) | Interpretable multi-horizon model with covariates | fev-bench skill 18.37, below LightGBM (20.48) |

A 2026 re-benchmark of supervised LTSF models, titled "There are no Champions in Supervised Long-Term Time Series Forecasting", found no consistently best architecture. It attributes many claimed gains to non-standardised evaluation and biased comparisons. That is the right prior for architecture shopping.

### 2. Benchmark evidence and the baseline question

**What the leaderboards show.**
- **GIFT-Eval** (Salesforce, Oct 2024; continuously updated). It covers 23 datasets and 97 configurations across 7 domains, including "economy/finance" (mostly macro and sales-like series, not tick data), with MASE and CRPS normalised to seasonal naive. By late 2025 the leading zero-shot models scored about 0.70–0.75 normalised MASE (TimesFM-2.5 0.705, FlowState 0.726, Moirai-2 0.728, Toto-1 0.75). That is roughly a 25–30% error reduction versus seasonal naive. As of 9 April 2026 Chronos-2 ranked first among pretrained models, though it and TTM-R3 were trained partly on GIFT-Eval training data. That makes their "zero-shot" scores "not entirely rigorous" for some datasets, in the authors' words.
- **fev-bench** (Amazon, Sep 2025; live leaderboard). It has 100 tasks, 46 with covariates and 35 multivariate, with bootstrapped confidence intervals. The July 2026 snapshot gives these MASE skill scores against seasonal naive:
  - Chronos-2: 35.5
  - TiRex-2: 33.74
  - Toto-2.0-2.5B: 32.54
  - TabPFN-TS-3: 30.56
  - TimesFM-2.5: 30.2
  - LightGBM: 20.48
  - TFT: 18.37
  - PatchTST: 17.52
  - DeepAR: 16.44
  - statistical ensemble: roughly 11–16
  - naive: below 0 (worse than seasonal naive)
  
  fev-bench explicitly "prioritizes broad coverage… over fully leakage-free evaluation" and relies on *self-reported* training overlap. The authors point to Impermanent and TS-Arena as complementary leakage-free protocols. The TIME benchmark (2026) is marketed as contamination-resistant.
- **A telling result.** Google's automated code-search system ERA (Sep 2025) beat the entire May 2025 GIFT-Eval leaderboard with per-dataset solutions that "showed strong convergence towards gradient boosting and ensemble/decomposition models". Well-engineered boosted trees remain extremely competitive even on the benchmarks foundation models dominate.

**The critiques, and where evidence is weak or contested.**
- *"Are Transformers Effective for Time Series Forecasting?"* (Zeng et al., AAAI 2023) showed DLinear beating Informer, Autoformer and FEDformer on the LTSF datasets (ETT, Electricity, Traffic, Weather, ILI, Exchange). PatchTST and iTransformer were partly responses to it. The lasting lesson is that linear baselines are mandatory, not that Transformers never work.
- *LTSF benchmark validity.* Hewamalage et al. (2023) and Bergmeir (2024) argue that the LTSF datasets are over-represented and similar, that fixed-horizon protocols are unrealistic, and that some series (notably Exchange-rate) are close to random walks, where naive forecasts are hard to beat. Roque et al. (2025) show small gains can "vanish or even reverse with minor benchmark changes". QuitoBench (2026) notes that 50% of GIFT-Eval series have fewer than 200 points and flags direct and indirect leakage channels.
- *Deep models vs boosted trees on tabular features.* The tabular-ML literature (e.g. Grinsztajn et al., NeurIPS 2022) and forecasting competitions (M5, Kaggle retail) consistently favour GBDTs on engineered, heterogeneous features. In finance specifically, Rahimikia et al. find CatBoost/LightGBM the strongest benchmarks for daily returns. In its competition wrap-up, G-Research said all three top finishers of its crypto competition (Meme Lord Capital, Nathaniel Maddux and GABA) used LightGBM, and that "feature engineering… contributed most to their wins." The counter-evidence is TabPFN-style tabular foundation models, which are competitive on fev-bench but slow. It has not been validated on noisy financial targets.
- **Net:** on *forecastable* series, TSFMs now clearly beat seasonal naive, ARIMA/ETS and per-dataset deep nets, and usually beat untuned LightGBM. How big their margin over well-tuned GBDTs with good features is remains contested, and it depends on leakage.

### 3. What changes on financial/trading data

**Foundation models on returns (independent evidence).**
- **Rahimikia, Ni & Wang, "Re(Visiting) TSFMs in Finance" (arXiv, Nov 2025).** They use daily excess returns over 34 years in 94 countries (about 2 billion observations), with 2001–2023 U.S. out-of-sample testing. Key numbers:
  - CatBoost's R²_OOS is −0.03% at a 512-day window, with directional accuracy of 51.16%.
  - Zero-shot Chronos-large reaches R²_OOS −1.37%. TimesFM-500M reaches −2.80%, with directional accuracy just below 50%.
  - Long–short decile portfolios before costs: benchmark Sharpe 6.46, Chronos-large 2.92, TimesFM-2 −0.18.
  - Fine-tuned TSFM portfolios have Sharpe between −3.34 and 0.07.
  - Pretraining Chronos-small from scratch on financial data raises Sharpe to 5.42. With synthetic augmentation it reaches 51.74% directional accuracy against CatBoost's 51.16%.
  
  Note that every number here is gross of costs, daily and cross-sectional, a very different problem from single-asset minute direction.
- **Noguer i Alonso & Pereira Franklin (arXiv, Jun 2026).** They forecast 20-business-day returns for AAPL, AMZN, GOOG, JPM and META. Zero-shot TSFMs won 8 of 10 tasks against NBEATS, NHITS, PatchTST, iTransformer and KAN. But "gains over the random-walk benchmark are small and sparse". Even the *winning* model underperformed a zero-return random walk on AAPL, JPM and META. Only Chronos on AMZN and Moirai-2.0 on GOOG passed a one-sided Diebold–Mariano test (p = 0.0421 each, no multiplicity correction). The authors conclude TSFMs are "useful practical priors… but not… universal engines of statistically reliable alpha generation."
- **Interpretation for you.** A TSFM applied to BTC 1-minute prices or returns will mostly reproduce a near-martingale forecast with a good volatility envelope. Its quantile bands may be useful for *volatility/scale* forecasting. Its median is unlikely to contain directional information your TFT missed.

**Realistic edge magnitudes at minutes-to-hours.**
- Kitron & Wengrowicz, "Short-horizon mean reversion in cryptocurrency markets" (arXiv, Aug 2026; independent researchers, not yet peer-reviewed). This is the most methodologically careful recent crypto study found. It used 15-minute bars on 183 Binance spot pairs and 187 U.S. stocks/ETFs, a pre-fixed walk-forward, Benjamini–Hochberg FDR, a permutation null and a frozen six-month holdout. Results:
  - Mean crypto AUC was 0.531 (US equities 0.499), and BTC unthresholded accuracy was 52.3%.
  - The holdout gap attenuated from +0.031 to +0.020.
  - AUC after taker-flow-driven bars was 0.540, against 0.519 after flow-opposed bars.
  - The gross edge "peaks near 1.3 bp per trade against a 5 bp round-trip cost: large enough to detect, too small to clear benchmark spot capture costs."
  
  This is close to the ceiling you should expect from price and flow data at this horizon, and it is below costs for a taker.
- The G-Research crypto competition, scored by weighted Pearson correlation on short-horizon returns across 14 assets, produced winning correlations only a few hundredths above zero. For scale, one public Kaggle notebook ("G-Research Crypto Forecasting modelization") shows a private score of 0.0029.
- **Why published "high accuracy" results fail.** The main causes are:
  - overlapping labels and random k-fold splits (leakage)
  - features computed with future information (centred rolling windows, full-sample normalisation, survivorship)
  - evaluation on FI-2010, which is pre-normalised and "too simplistic, leaving ample space for models' overfitting"
  - reporting accuracy on imbalanced or flat-inclusive labels
  - ignoring spread, fees and slippage
  - selecting the best of many configurations without deflation
  
  LOBCAST (Prata et al., *Artificial Intelligence Review* 2024) reimplemented 15 deep LOB models. Papers claimed over 88% F1 on FI-2010, yet "all models exhibit a significant performance drop when exposed to new data".

**Order-flow / microstructure features: what is actually established.**
- **OFI.** Cont, Kukanov & Stoikov (*J. Financial Econometrics* 2014): over short intervals, price changes are "mainly driven by the order flow imbalance", with a linear relation whose slope is inversely proportional to depth. That relation is *contemporaneous* (10-second windows). Cont, Cucuringu & Zhang (*Quantitative Finance* 2023) showed that multi-level "integrated OFI" improves contemporaneous fit. They also found cross-asset OFI adds information for *intraday forecasting*, which is the predictive part, and it is much weaker. A 2026 Bayesian study reports out-of-sample R² rising from about 55% to about 80% with ten levels, again for contemporaneous price changes. Kolm, Turiel & Westray (2023) found deep nets trained on OFI outperform those trained on raw books, but predictability decays within seconds to minutes.
- **Crypto specifically.**
  - Anastasopoulos & Gradojevic (EFMA 2025) find aggregated order flow predicts daily and weekly crypto returns.
  - A 2026 preprint on Binance perpetual 1-second data (2022–Oct 2025) finds that order-flow imbalance, spreads and VWAP deviations are the most important GBDT features.
  - The Kitron & Wengrowicz result above shows that taker flow *conditions* short-horizon reversal.
  - Funding, open interest and liquidations are well motivated for longer horizons (hours to days) and for regime conditioning. No peer-reviewed evidence was found that they add robust *minute-level directional* skill net of costs.
- **Deep LOB models.** DeepLOB (Zhang, Zohren & Roberts, 2019) and its successors do show predictability. Briola et al. ("Deep Limit Order Book Forecasting", 2024) find it concentrated in *large-tick* instruments, and conclude that "high forecasting power does not necessarily correspond to actionable trading signals." BTC perps on Binance behave as a small-tick instrument relative to price, which is the less favourable case.

### 4. Methodology that matters more than architecture

| AFML tool | What it does | Independent support | Verdict |
|---|---|---|---|
| **Purged / embargoed CV, CPCV** | Removes train samples whose labels overlap the test window | Strong logic. Arian, Norouzi & Seco (*Knowledge-Based Systems* 2024) found CPCV beat K-fold, purged K-fold and walk-forward on PBO and DSR, *in synthetic markets only*. Walk-forward remains the realism standard | **Adopt.** Keep walk-forward as the final test and add purging/embargo whenever labels span multiple bars |
| **Deflated Sharpe Ratio, PBO** | Corrects for selection among many trials and for non-normal returns | Bailey & López de Prado (*JPM* 2014); Bailey et al. (*J. Computational Finance* 2017); consistent with the Harvey–Liu multiple-testing literature | **Essential** for a cronjob that tries many feature sets. Log every trial |
| **Dollar/volume/tick bars** | Sample by activity, not clock | Mixed. A 2026 frequency-controlled crypto study found better normality in some constructions but not uniformly (dollar-bar excess kurtosis 23.7 vs 50.4 depending on the pipeline). Practitioner replications find dollar bars "do not seem to yield lower serial correlation" in crypto. Better statistical properties are *not* evidence of better prediction | **Worth one pre-registered test.** Its main value is putting equal information into each sample during volatile and quiet periods |
| **Imbalance/runs bars** | Sample when signed flow deviates from expectation | Little independent evidence | Low priority |
| **Triple-barrier labels** | Label by which of profit-take, stop or time-out hits first, with volatility-scaled barriers | A Financial Innovation (2025) crypto study used information-driven bars with triple-barrier labels and deep learning on tick data (2018–2023). Other comparisons are mixed: in one MDPI (2025) multi-asset study, triple-barrier averaged a Sharpe of −0.03 across models | **Adopt for its framing.** Setting barriers above round-trip cost aligns the target with tradability |
| **Meta-labelling** | A secondary model decides whether and how much to act on a primary signal | Mostly vendor evidence. In Hudson & Thames' "Does Meta-Labeling Add to Signal Efficacy?" (Singh & Joubert, 2019), meta-labelling a Bollinger mean-reversion strategy lowered annualised return (44% vs 58%) but cut maximum drawdown from 24% to 12.3%. A widely repeated 17%→63% accuracy gain appears only in secondary summaries. Mechanically sound: it trades recall for precision, which helps when costs dominate | **Adopt experimentally,** with a very simple primary model (e.g. the reversal rule) |
| **Sample uniqueness weights, sequential bootstrap** | Down-weight overlapping labels | Theoretically correct; little independent quantification | Cheap; use the weights |
| **Fractional differentiation** | Make series stationary while keeping memory | Little independent evidence of forecasting gains at minute horizons | Low priority; returns plus levels-as-features work fine with GBDTs |

**Conformal prediction for "directional prediction with confidence intervals."**
- Standard split conformal assumes exchangeability, which time series violate. There are time-series variants:
  - **EnbPI** (Xu & Xie, ICML 2021) recycles ensemble residuals.
  - **ACI** (Gibbs & Candès, NeurIPS 2021) adapts the miscoverage level online to hit a long-run target under distribution shift.
  - **AgACI** (Zaffran et al., ICML 2022) aggregates several ACI learning rates.
  - **SPCI** (Xu & Xie, 2023) conditions on residual quantiles.
- Zaffran et al. found that EnbPI's "validity strongly depends on the data distribution", and that online calibration is "significantly better than offline".
- **What conformal can and cannot give you.** It gives *calibrated coverage* (for example, 90% of intervals contain the realised return over the long run). It does not give *edge*. For a binary up/down target, conformal classification returns {up}, {down} or {up, down}. On BTC minutes you should expect "{up, down}" nearly all the time. That is honest, and it doubles as a principled **abstention rule**: trade only when the set is a singleton. Coverage guarantees are marginal (averaged over time), not conditional on regime.
- **Practical path:** quantile LightGBM or a TSFM's quantile head, then conformalised quantile regression with ACI updates, then `P(return > cost)` or a singleton-set filter.

### 5. Practical tooling (as of late 2026)

- **Foundation-model access:** `chronos-forecasting` (≥2.1.0; Chronos-2 with pandas `predict_df` and fine-tuning), `timesfm` (2.5 with XReg covariates and LoRA example), `tirex` / TiRex-2, `toto` (GluonTS integration). AutoGluon-TimeSeries wraps Chronos with covariate regressors.
- **Nixtla:** `statsforecast` (fast ARIMA/ETS/Theta baselines), `mlforecast` (LightGBM/XGBoost with lag and rolling features and proper time-series CV), `neuralforecast` (PatchTST, iTransformer, TSMixer, NHITS, TFT and others behind one API). TimeGPT is a closed API.
- **GluonTS** (probabilistic models; used by fev and GIFT-Eval), **Darts** (unified API with backtesting and conformal wrappers), **sktime** (broad interface layer).
- **PyTorch Forecasting**, maintained under the sktime organisation. v1.7.0 was released 5 Apr 2026. A v2 API rework is on the 2026 roadmap, and TFT remains supported (recent releases added mixed precision and bug fixes). TFT is not abandoned, but it is also not where the field's progress is.
- **AFML tooling:**
  - **MLFinLab** (Hudson & Thames) is now under an "all rights reserved" commercial licence.
  - Open alternatives include **mlfinpy** (MIT) and **OpenQuant**, which is building triple-barrier/meta-labelling runbooks.
  - Many teams simply implement bars, triple-barrier, purged K-fold and DSR themselves (a few hundred lines) so they can audit them.
- **Conformal:** MAPIE (scikit-learn compatible; includes time-series methods such as EnbPI/ACI), `crepes`, or Darts' conformal models.
- **Sweeps / automation for a cronjob:**
  - **Optuna** with an SQLite/Postgres storage backend suits single-machine, resumable, cron-driven studies. It offers pruning, multi-objective support and a full trial history, and that history is exactly the N you need for the deflated Sharpe ratio.
  - Ray Tune is overkill on one laptop.
  - W&B sweeps add tracking but need network access. MLflow is a local alternative.
- **Data:** Binance's public archive provides tick-level trades and aggTrades plus klines for spot, USD-M and COIN-M. Futures archives also include funding and metrics files. This lets you extend well beyond one year and add ETH/SOL etc. for breadth.

### 6. Recommendation for Matthias and Dominik

**Will a foundation model beat the TFT?**
- **Zero-shot: almost certainly not on direction.** Run it anyway: one pre-registered test, zero-shot, costs about a day. Chronos-2 with your order-flow columns as past-only covariates, or TiRex-2, on the same walk-forward folds. It is a cheap, strong, *non-overfit* baseline that tells you whether your TFT was simply badly trained.
- **Fine-tuned: unlikely to help.** Rahimikia et al. found fine-tuning gave marginal gains and poor portfolios.
- **Where TSFMs could help:** volatility and range forecasting (to scale barriers and position sizes), and as embedding/feature extractors feeding a GBDT.

**Is LightGBM a stronger baseline? Yes. Make it the primary model.** Order-flow, funding and OI features are heterogeneous, tabular and noisy, which is exactly the regime where GBDTs win. They train in seconds on CPU, which makes a many-feature-set cronjob feasible. They also expose SHAP importances you can sanity-check against microstructure intuition.

**Is effort better spent on sampling and labelling? Yes. The target is the problem, not the network.** A concrete, pre-registerable plan:

1. **Establish the baseline ladder** (all on identical purged walk-forward folds, all scored net of costs):
   - coin flip / base rate
   - the Kitron–Wengrowicz style 3-parameter sign-reversal logit on 15-minute bars
   - LightGBM on your current features
   - Chronos-2 zero-shot with covariates
   
   If LightGBM cannot beat the reversal logit's out-of-sample AUC (~0.53) *and* its after-cost P&L, stop adding features and change the target.
2. **Change the target from "next-minute direction" to "tradable event outcome":**
   - Sample events with a CUSUM filter on dollar bars (e.g. ~50–200 bars/day).
   - Label with triple-barrier, with barriers set at k×local volatility and *at least* 2–3× round-trip cost (taker ~5 bp on spot per the 2026 study; check your actual perp fee tier and whether maker execution is realistic).
   - Horizons of 15 minutes to a few hours, not 1 minute.
3. **Meta-label a simple primary signal.** Examples: reversal after taker-flow-driven bars, or OFI/funding extremes. Let LightGBM predict whether the trade clears costs, and size or abstain accordingly. Conformalise the probabilities (ACI) so "confidence" means calibrated frequency.
4. **Expand data breadth, not just depth.** Use 3–5 years from the Binance archives and several liquid perps. With only one year of BTC, effectively one regime path, even a real edge cannot be distinguished from noise.
5. **Make the cronjob honest.**
   - Optuna study with every trial logged.
   - Frozen final holdout (e.g. the last 3–6 months) that the cronjob never sees.
   - Deflated Sharpe/PBO computed with the true trial count.
   - Benjamini–Hochberg across feature sets.
   - Pre-register the metric (after-cost Sharpe or AUC with a block-bootstrap CI) before each batch.
6. **Stopping rule.** Suppose that after steps 1–5 no configuration beats the reversal baseline net of costs on the frozen holdout. Then the evidence says the minute-horizon directional edge available to you is below costs. The better research questions become volatility forecasting, execution/maker strategies, funding-rate carry, or cross-sectional crypto (daily), where published edges are larger.

## Caveats

- **Recency and peer review.** Several key sources are 2025–26 arXiv preprints: Rahimikia et al., Noguer i Alonso & Pereira Franklin, Kitron & Wengrowicz, TiRex-2, Toto 2.0 and the bars comparison. TimesFM-3's benchmark claims come from Google's own blog with rank charts only. Vendor leaderboard claims (Chronos-2, Toto 2.0, TimesFM-3 each "#1") conflict because they rely on different benchmark snapshots and leakage rules.
- **Contamination.** Chronos-2 and TTM-R3 saw parts of GIFT-Eval's training data. fev-bench leakage is self-reported. Treat small leaderboard gaps (1–3 skill points) as ties.
- **Transfer gap.** Rahimikia et al. study daily cross-sectional equity returns and Noguer i Alonso 20-day single stocks. Neither is 1-minute BTC. The inference that TSFMs won't help at minute horizons is an extrapolation, strongly supported by near-efficiency but not directly tested.
- **Cost assumptions.** The 5 bp round-trip figure is for spot taker execution. Perp fees, VIP tiers, maker rebates and funding change the break-even. Recompute it for your own account before accepting or rejecting any signal.
- **AFML evidence.** CPCV's superiority was shown only in synthetic markets. Meta-labelling's large gains come mainly from a vendor. Bar-type benefits are construction-dependent. None of these is a proven source of alpha. They are ways to avoid fooling yourself and to frame tradable targets.