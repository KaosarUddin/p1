set.seed(123)  # Setting seed for reproducibility
n =200
p = 2

X = matrix(rnorm(n * p), ncol = p)

true_beta = c(1, -0.5)
pi = 1 / (1 + exp(-X %*% true_beta))
y = rbinom(n, 1, pi)

beta_hat = logistic_regression(X, y)
bootstrap_conf_intervals(X, y)
plot_logistic_curve(X, beta_hat, y)
confusion_matrix(X, y, beta_hat)
Prevalence(X, y, beta_hat)
Accuracy(X, y, beta_hat)
Sensitivity(X, y, beta_hat)
Specificity(X, y, beta_hat)
False_Discovery_Rate(X, y, beta_hat)
Diagnostic_Odds_ratio(X, y, beta_hat)
prevalencegrid(X, y, beta_hat)
