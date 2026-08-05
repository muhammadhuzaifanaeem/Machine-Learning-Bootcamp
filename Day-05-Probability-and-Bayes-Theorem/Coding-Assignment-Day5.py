# Coding

# 1. Implement Bayes' Theorem in Python.
# 2. Create a small dataset and estimate the probability of an event.
# 3. Build the medical screening probability calculator.

# Bayes' Theorem Example

prior = 0.01          # Probability of disease
likelihood = 0.99     # Probability of positive test if disease exists
evidence = 0.05       # Probability of positive test overall

posterior = (likelihood * prior) / evidence

print("Posterior Probability:", posterior)
