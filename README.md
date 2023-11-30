# Packages for Logistics regression

The package must contain the basic functions to perform logistic regression (e.g. estimate the coefficient vector β

which includes the independent variables/predictors plus the intercept) and obtain different outputs from the procedure. The estimator to be computed using numerical optimization is the following:

$$
\hat{\beta} := \arg\min_\beta \sum_{i=1}^n \left( -y_i \cdot \ln(p_i) - (1 - y_i) \cdot \ln(1 - p_i) \right),
$$
where
$$
p_i := \frac{1}{1 + \exp(-x_i^T \beta)},
$$
and $y_i$ and $x_i$ represent the \( i \)-th observation and row of the response and the predictors respectively.

The package is specifically designed to perform logistic regression using numerical optimization techniques. This package offers a suite of essential functions that enable users to execute logistic regression analysis and extract various key outputs from the process. The core functionalities include:

    Calculation of Initial Values for Optimization: Leveraging the least-squares method to provide starting points for the logistic regression optimization process.
    Bootstrap Confidence Intervals: Functionality to generate bootstrap samples and calculate confidence intervals, enhancing the robustness of the regression analysis.
    Logistic Curve Plotting: Tools to visualize the logistic regression curve, offering intuitive insights into the relationship between variables.
    Confusion Matrix Generation: Enables the construction of a confusion matrix, which is pivotal in evaluating the performance of the logistic regression model.
    Performance Metrics Computation: The package computes a range of performance metrics critical for assessing the efficacy of the logistic model. These include:
        Prevalence: Measures the proportion of the response variable in the dataset.
        Accuracy: Quantifies the overall correctness of the model.
        Sensitivity: Assesses the model's ability to correctly identify true positives.
        Specificity: Evaluates the model's accuracy in identifying true negatives.
        False Discovery Rate: Determines the proportion of false positives in the predicted positives.
        Diagnostic Odds Ratio: Offers a measure of the effectiveness of the logistic regression model.


Below are descriptions for each of the functions in  package. These descriptions provide an overview of what each function does, its parameters, and its return value.

**bootstrap_conf_intervals(X, y)**

Description:
Calculates bootstrap confidence intervals for logistic regression coefficients. This function employs bootstrap sampling to estimate the variability of the coefficient estimates, providing a range of plausible values for each coefficient.

Parameters:

    X: Matrix of predictors.
    Y: Binary response variable (0 or 1).

Returns:
Matrix of bootstrap confidence intervals for each coefficient.

**plot_logistic_curve(X, beta_hat, y)**

Description:
Plots the logistic regression curve. This visualization helps in assessing the fit of the logistic regression model to the observed data.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
A plot of the logistic curve against the data.

**confusion_matrix(X, y, beta_hat)**

Description:
Generates a confusion matrix for the logistic regression model. This matrix is a useful tool for summarizing the performance of a classification model.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
A confusion matrix detailing true positives, false positives, true negatives, and false negatives.

**Prevalence(X, y, beta_hat)**

Description:
Calculates the prevalence, the proportion of positive cases in the dataset, based on the logistic regression model predictions.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
The prevalence value.

**Accuracy(X, y, beta_hat)**

Description:
Computes the accuracy of the logistic regression model, reflecting the proportion of correct predictions.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
The accuracy value.


**Sensitivity(X, y, beta_hat)**

Description:
Calculates the sensitivity (true positive rate) of the logistic regression model, indicating how well the model identifies positive cases.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
The sensitivity value.

**Specificity(X, y, beta_hat)**

Description:
Measures the specificity (true negative rate) of the logistic regression model, showing the model's ability to identify negative cases.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
The specificity value.

**False_Discovery_Rate(X, y, beta_hat)**

Description:
Computes the false discovery rate, indicating the proportion of false positives in the positive predictions.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
The false discovery rate.

**Diagnostic_Odds_ratio(X, y, beta_hat)**

Description:
Calculates the diagnostic odds ratio, a measure of the effectiveness of the logistic regression model.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
The diagnostic odds ratio value.

**prevalencegrid(X, y, beta_hat)**

Description:
Generates a grid of prevalence values over a range of cutoffs, useful for understanding how the prevalence metric varies with different classification thresholds.

Parameters:

    X: Matrix of predictors.
    beta_hat: Estimated coefficients from the logistic regression.
    y: Binary response variable (0 or 1).

Returns:
A matrix of prevalence values for each cutoff.

**Check the Packages**

Now, we are going to apply the packages and justify whether we get the result if we apply with an example. 
```{r}
library(p1)
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
```
