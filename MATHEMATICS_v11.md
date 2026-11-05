# v11 mathematical formulation and validation scope

## 1. The unknowns and what the policy can observe

The *simulation-only* inertial truth is a ten-dimensional inertial vector about the mesh origin,

\[\pi=[m,mc_x,mc_y,mc_z,I_{O,xx},I_{O,yy},I_{O,zz},I_{O,xy},I_{O,xz},I_{O,yz}]^\top.\]

In analysis we transform it to the more interpretable physical vector

\[\vartheta=[m,c_x,c_y,c_z,I_{C,xx},I_{C,yy},I_{C,zz},I_{C,xy},I_{C,xz},I_{C,yz}]^\top,\]

using \(c=(mc)/m\) and \(I_C=I_O-m[(c^Tc)I-cc^T]\). The RL state consists of current estimates, proxy covariance entries, measured pose/velocity/angular rate, and the number of completed nudges. Neither the policy, reward, nor OED optimizer receives the true inertial vector. The truth is accessed only to compute post-hoc evaluation errors and coverage.

Use fixed physically meaningful scale \(D=\mathrm{diag}(500,0.05,0.05,0.05,55,55,55,25,25,25)\), with units consistent with \(\vartheta\), when constructing covariance eigenvectors. This avoids letting kilogram units dominate the spectral ordering of inertia uncertainty.

## 2. Dynamics and virtual impulse

Nudge action \(a_k\) specifies a point \(p\) on the convex asteroid hull, an inward-directed force vector \(F\), impulse \(J\), and pulse duration \(J/\|F\|\). The *training simulator* approximates the pulse as instantaneous at the current attitude. The central angular-momentum jump is

\[\Delta L_C=(p-c)\times J,\quad \Delta\omega = I_C^{-1}\Delta L_C,\]

followed by accurate torque-free rotational integration and center-of-mass translation.

This surrogate is **not** a robotic-contact or MuJoCo simulation, and does not include asteroid orbital gravity, contact compliance, spacecraft reaction, a CBF, or keep-in constraints. Those are explicitly outside the present study.

## 3. Continuously refreshed estimation, and covariance approximations

At *every sampled state*, the estimator recalculates its current ten-parameter solution, including the currently unfinished angular momentum segment. Only completed segments are committed to avoid repeatedly adding the same overlapping segment as independent evidence. The kinematic estimator uses a position/attitude regression against integrated force, with nuisance parameters initial CoM position and velocity, body-frame CoM, and reciprocal mass.

For the mass/CoM covariance, retain the full nuisance-state precision, invert to obtain the marginal covariance of \((c,q=1/m)\), and transform by the local Jacobian \(\partial(m,c)/\partial(c,q)\). For central inertia, use the six linear momentum coefficients to construct a local six-dimensional covariance. The two pieces are assembled into a block-diagonal ten-parameter proxy, then scaled by \(D^{-1}\).

**This is not a certified joint Bayesian posterior.** Mass and inertia share attitude measurement errors; the block approximation neglects their cross-covariances. The underlying kinematic regression treats samples as approximately independent, while angular pairs depend on imperfect measured states and impulse calibration. The physical-parameter transformation is local and Gaussian; weakly identified or strongly nonlinear regimes can violate its assumptions.

v11 also changes the v10 central inertia prior from origin inertia to properly parallel-axis-corrected central inertia, and places a finite local prior on reciprocal mass: \(\sigma_q=0.0006\,\mathrm{kg}^{-1}\), which near \(m=600\,\mathrm{kg}\) gives \(\sigma_m\approx216\,\mathrm{kg}\). This is an explicit modeling assumption requiring independent precontact justification; do not pick it by looking at test-set truth.

## 4. Local Gaussian expected information gain

For a candidate action, approximate measurement sensitivity as \(H(a)=\partial h(a,\vartheta)/\partial\vartheta\), with measurement noise covariance \(R\). Under a linear Gaussian model,

\[P_{k+1}(a)=\big(P_k^{-1}+H(a)^\top R^{-1}H(a)\big)^{-1}.\]

In code, kinematic nuisance parameters are retained while updating their larger precision; central inertia is updated through its angular regression. Both are then projected into scaled physical coordinates to form \(P_{k+1}(a)\). The **Gaussian EIG** is

\[U_D(a)=\tfrac12\log\frac{\det P_k}{\det P_{k+1}(a)}\quad\text{nats}.\]

The following **extra preference terms are not measured in nats**:

\[U_A(a)=\frac{\operatorname{tr}P_k-\operatorname{tr}P_{k+1}(a)}{\operatorname{tr}P_k},\]

\[U_{\rm weak}(a)=\frac{\operatorname{tr}[V_k^\top(P_k-P_{k+1}(a))V_k]}{\operatorname{tr}(V_k^\top P_k V_k)},\]

where the columns of \(V_k\) span the largest-eigenvalue eigenspace of \(P_k\). A subspace, rather than an arbitrary single eigenvector, is essential when the largest eigenvalue is repeated. The weighted utility is

\[U_{\rm total}(a)=U_D(a)+w_AU_A(a)+w_wU_{\rm weak}(a).\]

**Crucial improvement over v10:** a 30-second free-flight observation can reveal inertial information even if no nudge is applied. To avoid giving the nudge credit for passive information, v11 computes the information gain under a **zero-impulse counterfactual with the same horizon** and optimizes

\[\boxed{U_{\rm active}(a)=U_{\rm total}(a)-U_{\rm total}(a_{\rm passive})}.\]

The optimizer chooses the highest-scoring valid candidate from a seeded six-dimensional Sobol sequence at each of three decision points. The RL/SAC agent receives \(0.15U_{\rm active}(a)\) as its step reward. The scale 0.15 is a training convenience, **not** a physical unit or EIG.

If the weak information direction concentrates around, for example, \(I_{C,zz}\), candidate nudges with a larger \((p-c)\times J\) component about the body z-axis may become more informative. However, the highest torque is not automatically the best action: cross-parameter correlations, predicted motion sensitivity, measurement noise, and the remaining 30-second rotation all affect the predicted information.

## 5. Per-nudge and flight-relevant diagnostics

The research notebook logs for each of the three nudges:

1. The candidate's surface point, impulse direction, impulse magnitude and angular impulse \((p-c)\times J\).
2. Its total expected Gaussian EIG, passive expected EIG, and **incremental expected EIG** attributable to the nudge.
3. Its predicted posterior physical parameter standard deviations and weakest normalized eigenmodes.
4. The actual online estimator's **post-observation covariance log-det change**. This is an *observed covariance proxy*, not actual Bayesian information gain.
5. A one-step-ahead **prequential pose/velocity forecast** frozen before the force and evaluated only when new measurements arrive. The residual has no access to ground-truth inertia.
6. In simulation-only evaluation: final ten-component errors, marginal normalized errors, nominal 95% coverage, a nominal quadratic normalized error (not a certified NEES), and the mass/CoM/inertia norms.

Calibration seeds, RL teacher/training seeds, distillation seeds, and held-out final-evaluation seeds are disjoint. Comparisons share initial asteroid truth and per-nudge measurement-noise streams; action-dependent state and sample counts will still differ between policies.

## 6. Interpretation, limitations, and required validation

- A well-conditioned information matrix **does not imply unbiased identification**. Errors-in-variables, nonlinearities, and model mismatch can create bias.
- The local Gaussian covariance may be **overconfident**. Empirical coverage should be close to nominal over independent simulation truths; low coverage invalidates precision claims.
- A small final estimation error on a known simulated asteroid cannot be used for real-time nudge selection, because true parameters are unavailable on a real mission.
- SAC uses a neural stochastic policy over 3 decisions; it does not produce exact closed-form equations. The notebook optionally fits an episode-separated decision tree as an **approximate** piecewise policy, then checks its closed-loop errors.
- A one-step greedy Sobol OED policy, a trained SAC policy, and a hybrid candidate selector are **not** guaranteed to find a globally optimal three-nudge sequence.
- For eventual flight work, characterize colored and asynchronous optical-state errors, force-transmission uncertainty, time stamping, contact location uncertainty, CoM frame errors, environmental disturbances, and a high-fidelity orbital/contact backend before interpreting the covariance as a mission safety quantity.

## 7. File architecture

- `src/nudge_v11_information.py`: covariance transformation, eigenmodes, active-vs-passive EIG, Sobol OED.
- `src/nudge_v11_env.py`: Gymnasium three-nudge environment, continuous updates and predictive residual logs.
- `src/nudge_v11_translation.py`: v10 hybrid estimator with central-inertia-prior correction and explicit inverse-mass prior width.
- `src/asteroid_surface_v11.py`: finite/guarded convex-hull ray intersection.
- `src/nudge_v11_workflow.py`: common-seed evaluation, coverage, paired bootstrap, teacher data, CSV/JSON exports.
- `notebooks/three_nudge_rl_v11_uncertainty_guided.ipynb`: *all* notebook-controlled experiments; one `QUICK_MODE` boolean.
- `tests/test_nudge_v11.py`: numeric identities, physical model, geometry, estimator/Gym integration checks.
- `src/nudge_v10_*.py` and `model/asteroid.obj`: bundled reference dependencies for stand-alone use; these do not alter v10 unless manually copied over existing files.
