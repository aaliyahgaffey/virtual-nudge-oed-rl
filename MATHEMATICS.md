# Virtual-Nudge Inertial Identification: Mathematical Formulation

This document describes the mathematical model used to identify an asteroid's inertial properties through **sequential virtual nudges**, and the three approaches used to select those nudges: **optimal experimental design (OED)**, **reinforcement learning (RL)**, and a **hybrid OED–RL selector**.

The principal simulation uses an ideal-impulse, torque-free rigid-body surrogate. It is an active-identification study, **not** a model of spacecraft maneuvering, robotic contact, or flight-certified safety. The formulas below distinguish the implemented estimation and design approximations from quantities available only for simulation evaluation.

## 1. Problem formulation and notation

An episode consists of three sequential nudge decisions. After each nudge, the asteroid undergoes a nominal **30-second observation period**. New measurements update the parameter estimate before the next action is chosen.

Let $O$ be the asteroid mesh origin, $C$ its center of mass (CoM), $B$ the body frame, and $W$ the inertial/world frame. The ten inertial parameters defined about $O$ are

$$
\boldsymbol{\pi}
= \begin{bmatrix}
 m & h_x & h_y & h_z & I_{O,xx} & I_{O,yy} & I_{O,zz} & I_{O,xy} & I_{O,xz} & I_{O,yz}
\end{bmatrix}^{\!\top},
\qquad \boldsymbol{h}=m\boldsymbol{c},
$$

where $m$ is mass, $\boldsymbol{c}$ is the body-frame vector from $O$ to $C$, $\boldsymbol{h}$ is the first mass moment, and $\boldsymbol{I}_O$ is the symmetric inertia tensor about the mesh origin. For reporting and uncertainty evaluation, use the physical vector

$$
\boldsymbol{\theta}
= \begin{bmatrix}
 m & c_x & c_y & c_z & I_{C,xx} & I_{C,yy} & I_{C,zz} & I_{C,xy} & I_{C,xz} & I_{C,yz}
\end{bmatrix}^{\!\top}.
$$

The conversion is given by the parallel-axis theorem:

$$
\boldsymbol{c}=\frac{\boldsymbol{h}}{m},
\qquad
\boldsymbol{I}_C
=\boldsymbol{I}_O-m\left[(\boldsymbol{c}^{\top}\boldsymbol{c})\boldsymbol{I}_3
-\boldsymbol{c}\boldsymbol{c}^{\top}\right].
$$

The entries of $\boldsymbol{I}_C$ are **tensor components**, not necessarily the principal moments of inertia. The principal moments are its eigenvalues.

The hidden simulated value $\boldsymbol{\pi}_{\mathrm{true}}$ determines the dynamics. It is **not** given to the estimator, OED planner, RL observation, or reward calculation; it is used only for post-hoc evaluation.

## 2. Asteroid dynamics and virtual impulses

The measured rigid-body state is

$$
\boldsymbol{x}(t)=\left(\boldsymbol{r}_O^W,\ \boldsymbol{v}_O^W,\
\boldsymbol{R}_{WB},\ \boldsymbol{\omega}_B\right),
$$

where $\boldsymbol{R}_{WB}$ rotates body-frame vectors into world coordinates and $\boldsymbol{\omega}_B$ is the body-frame angular velocity.

A virtual nudge is defined by an application point $\boldsymbol{p}_B$ on the asteroid hull, a unit push direction $\widehat{\boldsymbol{f}}_B$, a scalar impulse $J$, and force magnitude $F$. Its body-frame linear impulse and central angular impulse are

$$
\boldsymbol{J}_B=J\widehat{\boldsymbol{f}}_B,
\qquad
\Delta\boldsymbol{L}_{C,B}
=(\boldsymbol{p}_B-\boldsymbol{c})\times\boldsymbol{J}_B.
$$

For an instantaneous impulse applied at the current attitude,

$$
\Delta\boldsymbol{v}_C^W
=\frac{\boldsymbol{R}_{WB}\boldsymbol{J}_B}{m},
\qquad
\Delta\boldsymbol{\omega}_B
=\boldsymbol{I}_C^{-1}\Delta\boldsymbol{L}_{C,B}.
$$

The corresponding commanded pulse duration is $T_p=J/F$. **The surrogate applies the impulse instantaneously** rather than integrating a finite-duration contact force. The nominal pulse duration remains part of the commanded action and observation horizon.

Between impulses, the asteroid follows force-free translation and torque-free rotation:

$$
\ddot{\boldsymbol{r}}_C^W=\boldsymbol{0},
\qquad
\boldsymbol{I}_C\dot{\boldsymbol{\omega}}_B
+\boldsymbol{\omega}_B\times(\boldsymbol{I}_C\boldsymbol{\omega}_B)
=\boldsymbol{0},
$$

$$
\dot{\boldsymbol{R}}_{WB}
=\boldsymbol{R}_{WB}[\boldsymbol{\omega}_B]_{\times},
\qquad
\boldsymbol{r}_O^W
=\boldsymbol{r}_C^W-\boldsymbol{R}_{WB}\boldsymbol{c}.
$$

Here $[\boldsymbol{\omega}]_{\times}$ is the cross-product matrix. The numerical surrogate integrates rotational motion and quaternion kinematics to predict the next observed state.

## 3. Nudge action parameterization and admissibility

The planner and RL agent share a continuous **six-dimensional** action,

$$
\boldsymbol{a}\in[-1,1]^6.
$$

It encodes (1) hull longitude, (2) a hull-ray vertical coordinate, (3) tangent-plane azimuth, (4) tangential push fraction, (5) impulse magnitude, and (6) force magnitude. The surface geometry maps the first two coordinates to $\boldsymbol{p}_B$ and an outward normal $\widehat{\boldsymbol{n}}$.

With orthogonal surface tangent vectors $\widehat{\boldsymbol{t}}_1$ and $\widehat{\boldsymbol{t}}_2$, the inward-dominated push direction is

$$
\widehat{\boldsymbol{f}}_B
=-\sqrt{1-\beta^2}\,\widehat{\boldsymbol{n}}
+\beta\left(\cos\phi\,\widehat{\boldsymbol{t}}_1
+\sin\phi\,\widehat{\boldsymbol{t}}_2\right),
\qquad 0\le\beta\le 0.2.
$$

An action must satisfy actuator limits of the form

$$
J_{\min}\le J\le J_{\max},\qquad
F_{\min}\le F\le F_{\max},\qquad
T_{\min}\le\frac{J}{F}\le T_{\max},
$$

$$
\left\|\boldsymbol{p}_B\times
\left(F\widehat{\boldsymbol{f}}_B\right)\right\|
\le\tau_{\max}.
$$

The last bound is computed **about the mesh origin** in the action mapper, whereas the angular response depends on the lever arm about the **center of mass**. Default actuator settings in the current surrogate are:

| Quantity | Default |
| --- | ---: |
| Impulse range $J$ | $0.2$–$2.0\ \mathrm{N\,s}$ |
| Force range $F$ | $1.0$–$4.5\ \mathrm{N}$ |
| Maximum torque about mesh origin | $8.0\ \mathrm{N\,m}$ |
| Pulse-duration range $T_p$ | $0.12$–$2.0\ \mathrm{s}$ |
| Maximum tangential fraction $\beta$ | $0.2$ |
| Observation duration after a nudge | $30\ \mathrm{s}$ |

These are **actuator/admissibility restrictions**, not a guarantee of asteroid containment. The surrogate does **not** currently impose an explicit asteroid drift, speed, spacecraft collision, or control-barrier-function (CBF) constraint. Such limits must be added and validated before they can be claimed as enforced safety requirements.

## 4. Online inertial-parameter estimation

The estimator combines two complementary regressions: a **kinematic regression** for mass and CoM, and an **angular-momentum regression** for central inertia. This combination is sometimes called the *hybrid estimator*; it is distinct from the *hybrid OED–RL action selector* described later.

### 4.1 Mass and center of mass from pose history

For known applied world-frame force $\boldsymbol{F}^W(t)$ and reciprocal mass $q=1/m$,

$$
\boldsymbol{r}_O^W(t)
=\boldsymbol{r}_{C,0}^W
+\boldsymbol{v}_{C,0}^W(t-t_0)
-\boldsymbol{R}_{WB}(t)\boldsymbol{c}
+q\boldsymbol{K}(t),
$$

where

$$
\boldsymbol{K}(t)
=\int_{t_0}^{t}(t-s)\boldsymbol{F}^W(s)\,ds.
$$

Under the ideal-impulse approximation, with world-frame impulses $\boldsymbol{J}_j^W$ at times $t_j$,

$$
\boldsymbol{K}(t)
=\sum_{t_j\le t}(t-t_j)\boldsymbol{J}_j^W.
$$

This gives a linear regression with nuisance parameters $\boldsymbol{r}_{C,0}^W$ and $\boldsymbol{v}_{C,0}^W$:

$$
\boldsymbol{r}_O^W(t)
=\underbrace{\begin{bmatrix}
\boldsymbol{I}_3 & (t-t_0)\boldsymbol{I}_3 & -\boldsymbol{R}_{WB}(t) & \boldsymbol{K}(t)
\end{bmatrix}}_{\boldsymbol{H}_t}
\underbrace{\begin{bmatrix}
\boldsymbol{r}_{C,0}^W\\
\boldsymbol{v}_{C,0}^W\\
\boldsymbol{c}\\q
\end{bmatrix}}_{\boldsymbol{z}}.
$$

A regularized weighted least-squares estimator updates $\widehat{\boldsymbol{z}}$ using measured position, attitude, and impulse history. Then

$$
\widehat{m}=\frac{1}{\widehat{q}},
\qquad
\widehat{\boldsymbol{h}}
=\widehat{m}\,\widehat{\boldsymbol{c}}.
$$

Mass requires sufficient known force excitation; CoM estimation also depends on attitude variation and geometric observability.

### 4.2 Central inertia from angular momentum

For a torque-free interval interrupted by a known impulse, angular-momentum balance about the CoM gives

$$
\boldsymbol{R}_b\boldsymbol{I}_C\boldsymbol{\omega}_{B,b}
-\boldsymbol{R}_a\boldsymbol{I}_C\boldsymbol{\omega}_{B,a}
\approx\boldsymbol{L}_O
-\boldsymbol{r}_C^W(t_{\mathrm{imp}})\times\boldsymbol{J}^W,
$$

where $\boldsymbol{L}_O$ is the accumulated world-frame angular impulse about the world origin and the CoM correction is evaluated at the impulse. This is exact for the ideal instantaneous impulse when the reference quantities are consistent; for a finite pulse, the time-varying lever arm must be integrated.

Write the symmetric central inertia using six basis matrices $\boldsymbol{E}_i$:

$$
\boldsymbol{I}_C
=\sum_{i=1}^{6} \iota_i\boldsymbol{E}_i,
\qquad
\boldsymbol{A}_{I}[:,i]
=\boldsymbol{R}_b\boldsymbol{E}_i\boldsymbol{\omega}_{B,b}
-\boldsymbol{R}_a\boldsymbol{E}_i\boldsymbol{\omega}_{B,a}.
$$

The six inertia coefficients are estimated from

$$
\boldsymbol{A}_I\boldsymbol{\iota}
\approx\boldsymbol{L}_O
-\boldsymbol{r}_C^W(t_{\mathrm{imp}})\times\boldsymbol{J}^W,
$$

using regularization and physically admissible inertia checks. A positive-definite $\boldsymbol{I}_C$ must also satisfy the principal-moment triangle inequalities.

### 4.3 Measurement-by-measurement updates

The estimator refreshes its parameter solution at **every incoming sampled state**, including a preview of the current unfinished angular-momentum interval. Completed nonoverlapping intervals are committed; the unfinished interval is replaced by a new preview, **not repeatedly counted as independent data**.

Thus *continuous estimation* here means measurement-by-measurement estimation at finite simulation/sensor cadence. There are still only **three discrete action decisions**.

## 5. Approximate parameter uncertainty

The kinematic fit retains the full precision matrix for its ten coefficients before extracting the marginal covariance of $(\boldsymbol{c},q)$. A local Jacobian converts that covariance into physical $(m,\boldsymbol{c})$ coordinates. Separately, the six-coefficient angular regression supplies a local covariance for $\boldsymbol{I}_C$.

These are combined into a **block-diagonal design covariance proxy**; shared attitude errors and cross-correlations between mass/CoM and inertia are not fully represented. The proxy is therefore **not a certified joint Bayesian posterior**.

For physically meaningful comparisons across differently dimensioned parameters, define

$$
\boldsymbol{D}
=\operatorname{diag}(500,\,0.05,\,0.05,\,0.05,\,
55,\,55,\,55,\,25,\,25,\,25),
$$

with entries carrying the units of the corresponding components of $\boldsymbol{\theta}$. The normalized covariance is

$$
\boldsymbol{P}_k
=\boldsymbol{D}^{-1}\boldsymbol{P}_{\theta,k}\boldsymbol{D}^{-1}.
$$

A generic linear-Gaussian observation model with sensitivity $\boldsymbol{H}(a)$ and noise covariance $\boldsymbol{R}$ would predict

$$
\boldsymbol{P}_{k+1}(a)
=\left[\boldsymbol{P}_{k}^{-1}
+\boldsymbol{H}(a)^{\top}\boldsymbol{R}^{-1}\boldsymbol{H}(a)\right]^{-1}.
$$

**Implementation detail:** the code actually updates separate kinematic nuisance-state and angular-inertia precision matrices under candidate predicted trajectories, then projects the corresponding covariances into the normalized ten-dimensional physical space. The compact equation above expresses the local-Gaussian design principle, not an exact full joint update.

## 6. Optimal experimental design (OED)

At decision $k$, OED uses the **current measured state and parameter estimate** to predict how each admissible action would change the uncertainty proxy over the next observation horizon.

### 6.1 D-optimal, A-optimal, and weak-mode terms

Let $\boldsymbol{P}_k$ be the current normalized covariance and $\boldsymbol{P}_{k+1}(a)$ its model-predicted value after action $a$. The local-Gaussian D-optimal information-gain term is

$$
U_D(a)
=\frac{1}{2}\log\frac{\det\boldsymbol{P}_k}
{\det\boldsymbol{P}_{k+1}(a)}.
$$

Under a calibrated linear-Gaussian model, $U_D$ is an expected information gain in **nats**. The implementation also includes two normalized design preferences:

$$
U_A(a)
=\frac{\operatorname{tr}(\boldsymbol{P}_k)
-\operatorname{tr}(\boldsymbol{P}_{k+1}(a))}
{\operatorname{tr}(\boldsymbol{P}_k)},
$$

$$
U_{\mathrm{weak}}(a)
=\frac{\operatorname{tr}\!\left[\boldsymbol{V}_k^{\top}
\left(\boldsymbol{P}_k-\boldsymbol{P}_{k+1}(a)\right)
\boldsymbol{V}_k\right]}
{\operatorname{tr}\!\left(\boldsymbol{V}_k^{\top}\boldsymbol{P}_k\boldsymbol{V}_k\right)},
$$

where the columns of $\boldsymbol{V}_k$ span the largest-eigenvalue eigenspace of $\boldsymbol{P}_k$—the **least-constrained normalized parameter directions**. Using the whole tied eigenspace avoids choosing an arbitrary eigenvector when eigenvalues are repeated.

The combined score is

$$
U_{\mathrm{total}}(a)
=U_D(a)+w_A U_A(a)+w_W U_{\mathrm{weak}}(a).
$$

The default weights for this uncertainty-aware score are $w_A=w_W=1$. **The trace and weak-mode terms are not measured in nats**; they are dimensionless preferences added to the D-optimal term.

### 6.2 Incremental information from active excitation

Observation during free flight can improve inertial estimates **even without a new nudge**. To avoid attributing passive information to the action, compare each candidate to a **zero-impulse counterfactual** with the same total time horizon and sampling assumptions:

$$
\boxed{
U_{\mathrm{active}}(a)
=U_{\mathrm{total}}(a)
-U_{\mathrm{total}}(a_{\mathrm{passive}})
}.
$$

Here the passive action represents observation of the currently predicted free motion without another impulse. The active score is therefore the model-predicted **incremental value of the nudge** above continued observation alone.

### 6.3 Sequential OED policy

A seeded six-dimensional Sobol sequence generates a finite set $\mathcal{A}_k^{\mathrm{cand}}$ of candidate controls. Each is converted to an admissible nudge and scored using the current estimate:

$$
\boxed{
\boldsymbol{a}_k^{\mathrm{OED}}
=\underset{\boldsymbol{a}\in\mathcal{A}_k^{\mathrm{cand}}}{\arg\max}
\ U_{\mathrm{active}}(\boldsymbol{a})
}.
$$

This is a **greedy, closed-loop finite-candidate optimization**; it is not an analytical global solution over all possible three-nudge sequences. After each observation interval, the estimator and OED search are updated before the next decision.

## 7. Reinforcement learning (RL)

The RL policy uses the same action parameterization and rigid-body environment as OED, allowing comparable evaluations. The current training approach uses **Soft Actor-Critic (SAC)**, which learns a stochastic continuous-action policy rather than a symbolic control law.

The policy observation has **84 normalized entries**, consisting of:

- 10 current normalized inertial-parameter estimates;
- 10 transformed covariance-diagonal entries and 45 covariance correlations;
- 3 position components, 3 velocity components, 9 attitude-matrix entries, and 3 angular-rate components;
- 1 normalized nudge counter.

The current observation does **not** reveal the simulator's hidden true inertial parameters.

For a valid action, the environment reward is the predicted incremental design utility scaled for training:

$$
r_k=0.15\,U_{\mathrm{active}}(\boldsymbol{a}_k).
$$

The factor $0.15$ is a numerical training scale, **not** a physical unit or conversion of nats. Invalid actions or numerical failures can terminate an episode with a configured failure reward (default $-10$).

SAC learns a policy $\pi_{\mathrm{RL}}(\boldsymbol{a}\mid\boldsymbol{o})$ to increase expected cumulative reward over the **three decision steps**, with entropy regularization characteristic of SAC. Unlike greedy OED, the learned policy may account implicitly for the effect of current choices on later states and decisions; **it is not guaranteed** to discover a globally optimal sequence or to produce interpretable closed-form rules.

## 8. Hybrid OED–RL action selection

The **hybrid policy** combines two competing candidate actions at each nudge decision:

$$
\boldsymbol{a}_k^{\mathrm{OED}}
=\operatorname{OED}(\widehat{\boldsymbol{\theta}}_k,\boldsymbol{P}_k,\boldsymbol{x}_k),
\qquad
\boldsymbol{a}_k^{\mathrm{RL}}
=\pi_{\mathrm{RL}}(\boldsymbol{o}_k).
$$

Both proposals are evaluated using the **same model-based incremental OED utility**, computed before executing the nudge:

$$
\boxed{
\boldsymbol{a}_k^{\mathrm{hybrid}}
=\underset{\boldsymbol{a}\in
\{\boldsymbol{a}_k^{\mathrm{OED}},\,\boldsymbol{a}_k^{\mathrm{RL}}\}}
{\arg\max}\ U_{\mathrm{active}}(\boldsymbol{a})
}.
$$

The action with the larger predicted score is executed. Ties favor OED in the current selector. After the resulting observation interval, the shared estimator updates and the OED, RL, and hybrid strategies can make their next decisions from the updated state.

This is a **candidate-selection hybrid**, not a neural policy constrained by an OED optimizer, not a weighted average of the two actions, and not a guarantee of physical safety. It may outperform either proposal under the **chosen surrogate information score**, but greater predicted score does not automatically imply smaller realized estimation error.

The term *hybrid estimator* elsewhere in this document instead refers to combining the kinematic and angular-momentum identification regressions. Those are two distinct uses of the word *hybrid*.

## 9. Evaluation and validation

Evaluate **random**, **greedy OED**, **SAC/RL**, and **hybrid OED–RL** policies under matched simulation configurations and held-out random seeds. Initial asteroid truth and measurement-noise streams should be coordinated between policies where possible, although action-dependent trajectories and sample counts can differ.

Useful per-nudge quantities include:

1. Selected surface point, inward push direction, impulse magnitude, commanded force and angular impulse $(\boldsymbol{p}_B-\widehat{\boldsymbol{c}})\times\boldsymbol{J}_B$.
2. Predicted total Gaussian EIG, passive-observation EIG, and **incremental EIG above passive observation**.
3. Predicted physical-parameter standard deviations, covariance eigenvalues and weak-mode directions.
4. Actual **change in the estimator's covariance proxy** over the observation interval; this is **not** measured mutual information or a certified posterior entropy reduction.
5. A prequential pose/velocity/attitude forecast frozen before action execution and scored against measurements acquired afterward.

For **simulation-only** evaluation, compare the final estimate to the hidden truth. Example error measures include

$$
e_m=|\widehat m-m_{\mathrm{true}}|,
\qquad
e_c=\left\|\widehat{\boldsymbol{c}}-\boldsymbol{c}_{\mathrm{true}}\right\|_2,
\qquad
e_I=\left\|\widehat{\boldsymbol{I}}_O-\boldsymbol{I}_{O,\mathrm{true}}\right\|_F.
$$

For uncertainty calibration, report standardized component errors

$$
z_i
=\frac{\widehat\theta_i-\theta_{i,\mathrm{true}}}
{\sqrt{(\boldsymbol{P}_{\theta})_{ii}}},
$$

along with empirical rates of $|z_i|\le 1.96$ on independent simulated truths. Quadratic normalized-error statistics can be reported as **diagnostic proxies**, not as certified consistency tests, because the covariance is approximate.

Use paired policy comparisons, independent evaluation seeds, and explicit success/failure rates. Small final estimation error on a particular simulated asteroid is not evidence that the uncertainty approximation is calibrated across asteroid properties and measurements.

## 10. Assumptions and limitations

- **Dynamics:** The primary training/design backend assumes free translation, ideal instantaneous impulses, rigid-body motion, and torque-free post-impulse rotation. It does not model orbital perturbations or finite-duration contact compliance.
- **Actuation:** The virtual force has no equal-and-opposite servicer reaction. Feasibility of spacecraft/robot placement and real contact transmission must be established separately.
- **Safety:** The action mapper enforces configured force, impulse, pulse-duration, tangential-direction and nominal torque limits. It does **not** establish asteroid drift containment, collision avoidance, or CBF safety.
- **Identification:** The kinematic regression and inertia regression share noisy attitude information. Local Gaussian covariance blocks omit relevant dependencies and may be biased or overconfident.
- **Planning:** Predicted information is evaluated under current estimates and an approximate measurement model. Weak observability, nonlinearities, force-calibration error and model mismatch can change realized results.
- **Optimization:** The Sobol OED policy is a finite-candidate greedy search, SAC is a learned approximation, and the hybrid compares only their two proposals. None is a proof of globally optimal experiment design.
- **Validation:** Reliable flight interpretation requires independent uncertainty calibration, realistic pose/velocity sensing, asynchronous and correlated measurement errors, uncertain contact location and impulse, high-fidelity rigid-body/force dynamics, and mission-specific motion constraints.

## 11. Implementation guide

The relevant implementation modules in this repository are:

| Mathematical component | File |
| --- | --- |
| Virtual rigid-body dynamics, inertial representation, action mapping | [`src/inertial_dynamics.py`](src/inertial_dynamics.py) |
| Online kinematic and angular-momentum estimator | [`src/translation_estimation.py`](src/translation_estimation.py) |
| Normalized covariance and active-information scoring | [`src/information_gain.py`](src/information_gain.py) |
| Sequential nudge environment and information-based reward | [`src/nudge_environment.py`](src/nudge_environment.py) |
| Shared OED, RL and hybrid policy evaluation | [`src/experiment_workflow.py`](src/experiment_workflow.py) |
| RL training entry point | [`scripts/train_virtual_nudge_rl.py`](scripts/train_virtual_nudge_rl.py) |
| Asteroid geometry mesh | [`model/asteroid.obj`](model/asteroid.obj) |

**Interpretation:** The mathematics specifies the current *research surrogate*, local estimation approximation, and decision objectives. It does not claim that optimizing these objectives certifies physical contact feasibility, an unbiased estimate, or spacecraft safety.
