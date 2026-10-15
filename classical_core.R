# =====================================================================
# classical_core.R : R version of the classical core (cahier sections 1, 2, 3.1, 3.2, 3.3)
# Uses the SAME data file as the Python notebook, so both languages work on identical data.
# Run the Python notebook first (it creates data/aligned_prices_returns.csv).
# install.packages(c("forecast","tseries","vars","urca","rugarch","FinTS"))
# =====================================================================
library(forecast); library(tseries); library(vars); library(urca); library(rugarch); library(FinTS)
set.seed(42)

# --- 1. Data: same period, same 90/10 chronological split ---------------
df <- read.csv("data/aligned_prices_returns.csv")
df$date <- as.Date(df$date)
df <- df[df$date >= as.Date("2019-01-01") & df$date <= as.Date("2026-10-14"), c("date", "gold_price", "btc_price")]
df <- na.omit(df)
df$lg <- log(df$gold_price); df$lb <- log(df$btc_price)        # log prices
df$rg <- c(NA, diff(df$lg)); df$rb <- c(NA, diff(df$lb))       # log returns
df <- df[-1, ]                                                  # drop the first row (no return), as in Python
n <- nrow(df); k <- floor(0.9 * n)
train <- df[1:k, ]; test <- df[(k + 1):n, ]
cat("n =", n, "| train =", nrow(train), "| test =", nrow(test), "\n")

# --- 2. Stationarity: ADF (H0 unit root) and KPSS (H0 stationary) -------
for (v in c("gold_price", "lg", "rg", "btc_price", "lb", "rb")) {
  x <- train[[v]]
  cat(sprintf("%-11s ADF p = %.4f | KPSS p = %.4f\n", v, adf.test(x)$p.value, kpss.test(x, null = "Level")$p.value))
}

# --- 3.1 ARIMA on log prices + residual diagnostics ----------------------
for (v in c("lg", "lb")) {
  fit <- auto.arima(train[[v]], seasonal = FALSE, ic = "aic")
  res <- residuals(fit)[-1]                      # drop the first value (d = 1: not a real residual)
  cat("\n", v, ": ", as.character(fit), "\n", sep = "")
  print(Box.test(res, lag = 10, type = "Ljung-Box"))   # independence
  print(jarque.bera.test(res))                         # normality
  print(ArchTest(res, lags = 10))                      # ARCH-LM
}

# --- 3.2 VAR on returns, Granger causality, cointegration ----------------
rets <- train[, c("rg", "rb")]
sel <- VARselect(rets, lag.max = 10, type = "const")
p <- max(sel$selection["AIC(n)"], 1)
var_fit <- VAR(rets, p = p, type = "const")
print(causality(var_fit, cause = "rg")$Granger)        # gold -> bitcoin
print(causality(var_fit, cause = "rb")$Granger)        # bitcoin -> gold
joh <- ca.jo(train[, c("lg", "lb")], type = "trace", ecdet = "const", K = 2)
print(summary(joh))                                    # if trace stat < critical value: no cointegration, no VECM

# --- 3.3 GARCH(1,1), Student-t errors, on returns (x100) -----------------
spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                   mean.model = list(armaOrder = c(0, 0)), distribution.model = "std")
for (v in c("rg", "rb")) {
  g <- ugarchfit(spec, data = train[[v]] * 100)
  print(coef(g))
  print(ArchTest(as.numeric(residuals(g, standardize = TRUE)), lags = 10))   # std. residuals must show no ARCH
  plot(train$date, sigma(g) / 100, type = "l", main = paste("GARCH(1,1) volatility:", v), ylab = "sigma", xlab = "")
}
