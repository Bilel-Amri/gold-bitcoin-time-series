# Gold vs Bitcoin — Time-Series Modelling, Forecasting and Diagnostics

**Mini-projet Séries Temporelles — TEK-UP — Master Data Science & IA**  
         Séries Temporelles pour Data Science & IA


| **Période des données** | 2019-01-01 → 2026-10-14 (date de fin fixée) |
| **Observations** | 1 955 jours ouvrés · train 1 759 (90 %) · test 196 (10 %) |

---

## 1. Objectif

Comparer l’or (ETF **GLD**) et le Bitcoin (**BTC-USD**) avec un pipeline complet de séries temporelles :

- diagnostics de stationnarité, ACF/PACF, STL, outliers ;
- modèles classiques : ARIMA, VAR + Granger, cointégration, GARCH(1,1)-t ;
- modèles avancés : LSTM, GRU (walk-forward), Prophet ;
- prévisions multi-horizon (h = 1, 5, 10) et test de Diebold-Mariano ;
- usage critique d’un LLM (hypothèses, vulgarisation, recommandations + audit).

**Question de modélisation.** Or et Bitcoin partagent-ils des dynamiques de court ou de long terme ? Quel modèle prévoit le mieux le log-prix, par rapport à la règle naïve « demain = aujourd’hui » ?

---

## 2. Structure du dépôt

```
gold vs btc/
├── notebook/
│   └── gold_bitcoin_project.ipynb    # Pipeline Python complet (livrable principal)
├── classical_core.R                  # Socle classique en R (même données, même split)
├── data/
│   ├── aligned_prices_returns.csv    # Données alignées (cache reproductible)
│   ├── arima_diagnostics.csv
│   ├── diebold_mariano.csv
│   ├── forecast_summary.csv
│   └── model_comparison_bitcoin.csv
├── figures/                          # Graphiques générés par le notebook
│   ├── 01_prices_returns.png
│   ├── 02_acf_pacf.png
│   ├── 03_acf_squared_returns.png
│   ├── 04_stl_gold.png / 04_stl_bitcoin.png
│   ├── 05_garch_volatility.png
│   ├── 06_loss_curves.png
│   ├── 07_rolling_corr.png
│   ├── 08_arima_fan.png
│   └── 09_rmse_ratio.png
├── llm/                              # Section LLM (4 points)
│   ├── prompt_1_hypotheses.txt
│   ├── prompt_2_vulgarization.txt
│   ├── prompt_3_recommendations.txt
│   └── ground_truth.csv
├── Rapport_Gold_Bitcoin_Series_Temporelles.docx
├── requirements.txt
└── README.md
```

---

## 3. Installation

### Python

```bash
python -m venv .venv
# Windows :
.venv\Scripts\activate
# Linux / macOS :
source .venv/bin/activate

pip install -r requirements.txt
```

Packages principaux : `yfinance`, `pandas`, `numpy`, `statsmodels`, `arch`, `pmdarima`, `tensorflow`, `prophet`, `scikit-learn`, `matplotlib`.

### R (socle classique uniquement)

```r
install.packages(c("forecast", "tseries", "vars", "urca", "rugarch", "FinTS"))
```

---

## 4. Exécution

### 4.1 Notebook Python (recommandé)

1. Ouvrir `notebook/gold_bitcoin_project.ipynb` dans Jupyter / VS Code / Cursor.
2. Vérifier que le répertoire de travail permet d’atteindre `../data`, `../figures`, `../llm`  
   (le notebook utilise `ROOT = Path("..").resolve()`).
3. Exécuter toutes les cellules de haut en bas (**Run All**).

Le notebook :

- télécharge (ou recharge depuis le cache) les données GLD et BTC-USD ;
- aligne les calendriers (jours ouvrés de l’or) ;
- produit EDA, ARIMA, VAR, GARCH, LSTM/GRU, Prophet, métriques et DM ;
- écrit les prompts LLM et la table de vérité dans `llm/`.

### 4.2 Script R

Depuis la **racine du projet** :

```r
setwd("C:/Users/.../gold vs btc/gold vs btc")   # adapter le chemin
source("classical_core.R")
```

Le script lit `data/aligned_prices_returns.csv` (créé par le notebook) et reproduit :

- tests ADF / KPSS ;
- ARIMA + diagnostics (Ljung-Box, Jarque-Bera, ARCH-LM) avec correction `residuals[-1]` ;
- VAR + Granger ;
- Johansen (cointégration) ;
- GARCH(1,1) Student-t.

---

## 5. Choix méthodologiques importants

| Choix | Justification |
|--------|----------------|
| **Date de fin fixée** (`2026-10-14`) | Reproductibilité : les résultats ne changent pas d’un jour à l’autre |
| **Jours ouvrés uniquement** | GLD ne cote pas le week-end ; pas de prix d’or inventés |
| **Outliers conservés** | Chocs de marché réels ; le GARCH en a besoin |
| **Résidus ARIMA : drop du 1er** | Avec `d = 1`, `residuals()[0]` n’est pas un vrai résidu |
| **Pas de VECM** | Pas de cointégration (Engle-Granger p ≈ 0,98 ; Johansen confirme) |
| **GARCH Student-t** | Queues épaisses (kurtosis Bitcoin ≈ 14) |
| **Walk-forward (pas de k-fold)** | Respect de l’ordre temporel |
| **Baseline naïve** | « demain = aujourd’hui » ; aucun modèle ne la bat de façon significative |

**Limite calendaire :** un rendement du lundi pour le Bitcoin couvre vendredi → lundi (3 jours).

---

## 6. Résultats principaux (échantillon test)

| Résultat | Détail |
|----------|--------|
| Stationnarité | Log-prix I(1) ; rendements stationnaires → `d = 1` |
| ARIMA | Or (0,1,0) ; Bitcoin (2,1,2) — résidus sans autocorrélation, mais ARCH + non-normalité |
| Granger | Aucune causalité (p = 0,73 et 0,94) |
| Cointégration | Absente (p ≈ 0,98) → pas de VECM |
| GARCH(1,1)-t | Persistance ≈ 0,96 (or) / 0,996 (Bitcoin) ; ARCH retiré (Bitcoin limite) |
| Prévision | **0 / 21** tests de Diebold-Mariano significatifs vs naïveté |
| Corrélation glissante 60 j | Maximum 0,653 le **2026-09-04** (pas mars 2020) |

**Recommandation opérationnelle**

- **Niveau de prix :** naïf / ARIMA(0,1,0)  
- **Risque :** GARCH(1,1)-t  
- **Lien or–Bitcoin :** pas de stratégie de paires ni de leading-lag fondée sur ces données  

---

## 7. Section LLM (4 points)

Ordre respecté (cahier des charges) :

1. Interprétation experte rédigée **avant** tout appel au LLM (section 6.2 du notebook).  
2. Trois prompts générés automatiquement à partir des statistiques réelles (`llm/prompt_*.txt`).  
3. Réponses du LLM collées en intégralité (6.4 et 6.6).  
4. Audit des erreurs / approximations avec la table `llm/ground_truth.csv` (6.5).

Aucun chiffre inventé par le LLM n’a été détecté ; les écarts relevés sont des approximations ou des omissions (Granger ≠ corrélation contemporaine, intervalles de l’or sous-couverts, pic de corrélation, etc.).

---

## 8. Dual langage (Python + R)

Le cahier exige le socle classique dans les deux langages.  
`classical_core.R` et le notebook Python utilisent le **même CSV** et le **même split 90/10**.  
Les conclusions (ordres ARIMA, Granger, absence de cointégration, diagnostics GARCH) coïncident.

Les parties avancées (LSTM, GRU, Prophet, évaluation multi-horizon, LLM) sont en Python uniquement, ce que le cahier autorise.

---

## 9. Reproductibilité

- Graine fixée : `SEED = 42` (Python et R).  
- Données mises en cache : `data/aligned_prices_returns.csv`.  
- Date de fin des téléchargements fixée (pas de `end = today`).  
- Toutes les conclusions chiffrées du notebook / du rapport sont **générées à partir des métriques calculées**, pas saisies à la main.

---

## 10. Livrables attendus

| Livrable | Fichier |
|----------|---------|
| Notebook exécuté | `notebook/gold_bitcoin_project.ipynb` |
| Code R (socle) | `classical_core.R` |
| Rapport 15–25 pages | `Rapport_Gold_Bitcoin_Series_Temporelles.docx` |
| Données & figures | dossiers `data/`, `figures/`, `llm/` |

---

## 11. Licence et usage

Projet académique — TEK-UP.  
Les prévisions et recommandations simulées **ne constituent pas un conseil d’investissement**.  
Données de marché via Yahoo Finance (`yfinance` / `quantmod`).
