# Wavelet-based density estimators in Python

This package implements the simple estimator based on average of wavelet functions
at sample points, and two shape-preserving based on estimating the square root
and then squaring it. The first using a pre-estimator of the square root and the
second based on k-nearest neighbours. In all cases, the estimators are for multivariate
data on the $[0,1]^d$ cube. If the data is not in that range, one can always
rescale.

## Simple wavelet estimator

Based on Ḧardle, W., Kerkyacharian, G., Picard, D. and Tsybakov, A., "Wavelets,
Approximation and Statistical Applications", Lecture Notes in Statistics, Springer,
New York, 1998.

## Estimator of square root from pre-estimator

Based on Pinheiro, Vidakovic, "Estimating the square root of a density via compactly
supported wavelets", 1997.

## Estimator of square root using nearest neighbours

Based on Aya, Geenes, Penev, "Shape-preserving wavelet-based multivariate density
estimation", 2017, Journal of Multivariate analysis

##

> $ python -m venv /path/to/env
> $ source /path/to/env/bin/activate
> $ python tests/something

