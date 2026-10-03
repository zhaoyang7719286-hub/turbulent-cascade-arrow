# Numerical definitions

Static results report `normalized_mean_Pi = mean(Pi)/std(Pi)` and `sign_bias_Pi = P(Pi > 0) - 1/2`. Accumulated statistics use nonoverlapping 5/10/20-frame windows in the recorded tables. Zero events distinguish the positive-event bias from the exact odd statistic `mean(sign(Pi))/2`.

The dynamic scalar is `X = mean(Pi)/mean(abs(Pi))`. Its recovery is

```text
R_Pi(t) = 1 - abs(X_branch(t) - X_natural(t)) / D0
D0 = abs(X_natural(0) - X_perturbed(0))
```

`D0` is fixed at the initial gap separately for each checkpoint and mask. Negative recovery is retained. Geometry recovery uses the analogous normalized `M` scalar and its own initial gap. No FRZ M trajectory is supplied.

`M = -tr(S^3) + omega_i S_ij omega_j/4`, `T = -tr(S^3)`, and `W/4 = omega_i S_ij omega_j/4`. In the generated triadic identity audit, a nonnegative capacity also has the historical label T; it is a different quantity.

Static block/mask counts are 32 primary and 24 independent-block masks. The independent comparator is the recorded `reference_star32`, not the unprojected raw DNS. The scale-8 raw-mean criterion fails: 8/24 raw means and 4/24 normalized means exceed their corresponding reference values. Mean and sign metrics are retained separately.

The six S3 panels show natural DNS count ratios and their algebraic sign-inverted counterparts. The dashed counterpart is not an empirical surrogate distribution.

The S6 candidate time is the natural initial `T_E = K(0)/epsilon(0)`. The recorded spread measure increases about 11.1% after this normalization. It does not establish improved collapse or universality.

For the phase-Haar ensemble, the expectation of any integrable odd flux functional is zero. Numerical checks support only the specified quadratic preservation and supplied implementation, not preservation of every other statistic. Dynamic conclusions refer to the tested NL/LIN/constructed FRZ protocol and selected scalar.
