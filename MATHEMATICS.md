# Sequential three-nudge inertial identification: mathematical formulation

This document is part of `sequential_nudge_rl_v9`. The formulae below describe **the implemented surrogate and local design approximation**, not a proof that a trained policy is globally optimal or safe in physical contact.

## Parameters and reference point

The 10 estimated quantities are

\[
\pi=[m,h_x,h_y,h_z,I_{O,xx},I_{O,yy},I_{O,zz},I_{O,xy},I_{O,xz},I_{O,yz}]^T,
\qquad h=m c_B.
\]

`O` is the asteroid's body-frame mesh origin; `c_B` is its center-of-mass vector expressed in body coordinates. The central inertia is

\[
I_C=I_O-m\big[(c_B^Tc_B)I_3-c_Bc_B^T\big].
\]

The six independent central-inertia components are estimated through a second linear regression, then transformed back into the **body-origin inertia** reported in the 10-vector. In particular, the diagonal elements of `I_O` are not necessarily principal moments.

## 1. Why the pose history identifies mass and center of mass

For an isolated free rigid body subject to known external **linear** force `F_W(t)` in world axes,

\[
\ddot r_C(t)=F_W(t)/m,
\qquad
r_O(t)=r_C(t)-R(t)c_B.
\]

Integrating force twice, with `q=1/m`, gives

\[
\boxed{r_O(t)=r_{C0}+v_{C0}(t-t_0)-R(t)c_B+q K(t)},
\qquad
K(t)=\int_{t_0}^t(t-s)F_W(s)\,ds.
\]

For a sequence of ideal impulses `J_j` applied at `t_j`,

\[
K(t)=\sum_{t_j\le t}(t-t_j)J_j.
\]

The unknowns `z=[rC0(3),vC0(3),cB(3),q]` enter **linearly**:

\[
\underbrace{r_O(t)}_{3\times 1}
=\underbrace{[I_3,\ (t-t_0)I_3,\ -R(t),\ K(t)]}_{H_t\in\mathbb R^{3\times 10}}z.
\]

The position and attitude measurements from *all* three observation periods contribute to weighted least squares, sampled at the configured sensor cadence. The solution is refreshed at each sample, while committed data are subsampled to avoid treating every overlapping estimate as statistically independent. This makes better use of the 30-second windows than simply subtracting two noisy velocity measurements.

After estimating `q>0` and `cB`, recover `m=1/q` and `h=m*cB`. The position history cannot identify mass if the force history lacks excitation. Likewise, the center of mass requires sufficient attitude changes; numerical rank or a small formal covariance alone does not prove accuracy.

The code constructs an online normal matrix using a design standard deviation for position/attitude data. Its covariance is a local Gaussian approximation and does not account for all cross-correlations or model errors.

## 2. Why CoM angular momentum identifies inertia

The angular momentum **about the center of mass** in world coordinates is

\[
H_C=R I_C\omega_B.
\]

For a time interval `[a,b]` with measured world-origin impulses `(J,L_O)`,

\[
R_b I_C\omega_{B,b}-R_a I_C\omega_{B,a}
\approx L_O-r_C(t_{\mathrm{imp}})\times J.
\]

The approximation uses the CoM at the impulse event; it is exact for the ideal instantaneous impulse and requires integral handling of finite-duration forces. In free flight, the right-hand side is zero.

Let `E_i` be the six symmetric 3×3 inertia basis matrices. The `3×6` angular regressor is

\[
A_I[:,i] =R_bE_i\omega_{B,b}-R_aE_i\omega_{B,a}.
\]

Solve `A_I i_C = L_C` with a weak prior and check positive definiteness and the principal-moment triangle inequality. Unlike the earlier world-origin 6D angular equation, this regression does not contain the large term `r_O×(m v_O)`, which amplifies velocity measurement noise when the origin lies several meters from the world origin.

**Physical caveat:** in the optional MuJoCo backend, the translation estimator registers measured finite-pulse linear force increments at their timestep midpoints, but the angular estimator presently uses the start-of-interval CoM when converting the entire angular impulse. This is accurate for a short ideal pulse, but only approximate for a finite pulse; validate its bias before treating it as a high-fidelity inference method.

## 3. Sequential information-gain objective: explicit OED rule

At nudge `k`, the current measurement history produces estimated parameters and two local Gaussian design precisions:

- `ΛT,k` for the 10 translation nuisance+target coefficients `[rC0,vC0,cB,q]`.
- `ΛI,k` for the 6 independent central-inertia coefficients.

For a candidate admissible action `a`, predict the asteroid free-flight states using the **current estimate**, not the hidden true parameters. Construct predicted design-information increments:

\[
\Delta\Lambda_T(a)=\sum_{t\in\mathcal{T}} H_t^T\Sigma_r^{-1}H_t
\qquad\text{(in normalized coordinates)},
\]

\[
\Delta\Lambda_I(a)=\sum_{j\in\mathcal{T}} A_{I,j}^T\Sigma_H^{-1}A_{I,j}
\qquad\text{(in normalized coordinates)}.
\]

For translation, the four scientifically relevant parameters are `u=[c_x,c_y,c_z,q]`; the initial position and velocity are nuisance parameters. Let `P_T=(ΛT,k)^-1` and `P_T^+(a)=(ΛT,k+ΔΛT(a))^-1`, and extract the 4×4 `u` marginal of each covariance. For angular inertia use the full 6×6 `P_I` and `P_I^+`.

Define, for any covariance block `P`:

\[
\mathcal{G}(P,P^+)
=\log\frac{\det P}{\det P^+}
+\alpha\frac{\operatorname{tr}P-\operatorname{tr}P^+}{\operatorname{tr}P}.
\]

Then the interpretable, greedy analytical nudge policy is

\[
\boxed{a_k^{\mathrm{OED}}=\underset{a\in\mathcal{A}}{\arg\max}
\left[\mathcal G(P_{T,u,k},P_{T,u,k}^+(a))+
\mathcal G(P_{I,k},P_{I,k}^+(a))\right]}
\]

with default `α=3`, selected from 64 seeded Sobol candidates. This is **explicit finite-candidate argmax**, not a closed-form continuum optimizer. It explicitly depends on information collected from the previous nudge through `ΛT,k`, `ΛI,k`, and the current state/parameter estimate.

The proposed reward is `0.15 ×` this expected objective (a positive scaling for SAC). This uses **predicted**, not realized, information; neither reward nor agent observation contains the simulator's true mass or inertia.

**Statistical caution:** the local Gaussian blocks treat translation and angular-inertia information as approximately independent, although they reuse pose measurements; their determinant reduction is an *engineering design score*, not guaranteed mutual information or a calibrated posterior entropy reduction.

## 4. Action parameterization and constraints

The continuous action `a∈[-1,1]^6` maps to:

1. A longitude on the convex asteroid hull.
2. A spherical `z` coordinate for the hull ray.
3. An azimuth in the surface tangent plane.
4. A tangential push fraction `β∈[0,0.2]`, with predominantly inward push.
5. A feasible impulse `J`.
6. A feasible contact force magnitude `F`.

The push direction is

\[
\hat f=-\sqrt{1-\beta^2}\hat n+
\beta(\cos\phi\hat t_1+\sin\phi\hat t_2).
\]

The pulse is `T_p=J/F`. The action map enforces force/impulse, torque `||p×F fhat||`, and pulse-duration bounds. The numerical surrogate does **not** certify CBF feasibility. The MuJoCo backend uses the existing spacecraft relative radial-shell CBF and collision guard.

**No asteroid displacement, linear-speed, or additional keep-in constraint or penalty is present**. Drift may be logged for diagnostics only.

## 5. RL and an interpretable approximation of the learned policy

Gymnasium state contains (i) current normalized 10-parameter estimate, (ii) its approximate normalized covariance diagonal and off-diagonal correlations, (iii) observed pose/twist, and (iv) remaining nudge count. The action is the six controls above. Each episode has three decisions. SAC is the suggested initial RL method.

An RL neural network does **not** generally provide a closed-form symbolic optimizer. After training, `--mode distill --teacher rl` fits a shallow regression tree to the agent's chosen actions, exports `approximate_policy_rules.txt`, and reports held-out action mean absolute error. The tree is a piecewise-constant **approximation of the trained policy**, not a derivation of the physical optimum.

The explicit formula in section 3 is the interpretable physics-based policy and a meaningful non-RL baseline; the learned policy is compared against this baseline with **matched simulator truth, initial conditions, seeds, noise, and three-nudge budget**.

## 6. Validation and limitations

- **Surrogate:** finite-duration force is replaced by a single instantaneous impulse. A high-precision quaternion rigid-body integration propagates subsequent motion.
- **Full MuJoCo:** the optional bridge executes finite-duration `mj_applyFT` pulses through the existing station-keeping spacecraft CBF and collision guard. It was not executable in the authoring environment.
- **Noise:** the program permits `sensor_sigma=0` or `1e-4` (the same sigma is used for simulated impulse measurement by default); zero-sigma runs inject no noise. A small variance floor in inference is a design regularizer, not injected noise.
- **Three nudges are not guaranteed to identify every parameter accurately.** Their success depends on observability, calibration, geometry, sensor quality, model fidelity, and initial angular motion.
- **No non-cooperative physical-contact guarantee:** virtual forces bypass robot–asteroid contact feasibility and equal/opposite reaction on the servicer. Physical execution must be separately established.
- **No certified Bayesian posterior:** uncertainty is a first-order engineering approximation; calibrate using repeated Monte Carlo runs, withheld data, and experiments.

## 7. Continuous estimation between nudge decisions

A fresh state measurement `y_j` arrives at time `t_j`. The estimator updates its **live** precision and information vector using a new *unfinished* independent interval `A_j`, `b_j`:

\[
  \widehat\pi(t_j) = \pi_0 + S\big(\Lambda_0 + \Lambda_{\rm committed} + A_j^T V_j^{-1} A_j\big)^{-1}
   \big(q_{\rm committed} + A_j^T V_j^{-1}(b_j-A_j\pi_0)\big).
\]

Only once the interval closes is its row/normal matrix committed; the live preview is then discarded. This prevents counting the *same overlapping interval* over and over. The kinematic block similarly recomputes a live fit at every sample. The angular block recomputes the inertia tensor from the committed **plus live-preview** angular segments whenever the mass/CoM block becomes numerically reliable. For the hybrid method, this is a **practical engineering estimator**, not a guaranteed unbiased or statistically calibrated 10-dimensional posterior.

The simulation and sensors operate with finite time resolution; thus “continuous” means **at each incoming measurement**, including all MuJoCo simulation substeps, rather than a single update after each nudge. The three decision events remain discrete.
