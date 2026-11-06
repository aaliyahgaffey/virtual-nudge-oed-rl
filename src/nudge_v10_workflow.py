"""All-in-notebook research functions for CBF-free sequential virtual nudge design.

No terminal subprocesses are used. Truth is used only in *evaluation*, not
by the policy/estimator/planner. All cross-policy comparisons rerun episodes
with the same initial random seed and dynamics distribution.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from src.nudge_v10_math import IDConfig, DesignConfig, action_to_nudge
from src.nudge_v10_env import ThreeNudgeEnv, EnvConfig
from src.nudge_v10_design import optimize_hybrid_one, predicted_hybrid_gain

OBS_NAMES = ([f"normalized_pi_{i}" for i in range(10)] +
             [f"log_cov_{i}" for i in range(10)] +
             [f"cov_corr_{i}_{j}" for i in range(10) for j in range(i+1,10)] +
             [f"r_{i}" for i in 'xyz'] + [f"v_{i}" for i in 'xyz'] +
             [f"R_{i}{j}" for i in range(3) for j in range(3)] +
             [f"omega_{i}" for i in 'xyz'] + ['nudge_fraction'])
ACTION_NAMES = ['longitude','surface_z','tangent_angle','tangent_fraction','impulse','force']
assert len(OBS_NAMES)==84 and len(ACTION_NAMES)==6


def make_env(root,icfg=None,dcfg=None,seed=None,record_trace=False,debug=True,sample_s=1.,randomized=True):
    env=ThreeNudgeEnv(root,
        id_config=icfg or IDConfig(sensor_sigma=1e-4),
        design_config=dcfg or DesignConfig(candidate_count=64),
        env_config=EnvConfig(backend='surrogate', randomized_truth=randomized,
            surrogate_sample_s=sample_s,debug_truth=debug,record_trace=record_trace))
    if seed is not None:env.reset(seed=int(seed))
    return env


def design_action(env,seed):
    """Explicit one-step optimal experiment design, no hidden true inertia."""
    a,p,utility,parts=optimize_hybrid_one(
        env.surface,env.est.last_state,env.est,env.dcfg,env.icfg,
        seed=int(seed*1009+env.n*311))
    return a,float(utility),parts


def predicted_action_utility(env,action):
    plan=action_to_nudge(action,env.surface,env.dcfg)
    value,_=predicted_hybrid_gain(env.est,env.est.last_state,plan,env.icfg,env.dcfg)
    return float(value)


def choose_action(policy,env,obs,seed,model=None,tree=None,rng=None,hybrid_margin=0.):
    """One decision between nudges; three decisions in total per rollout.

    hybrid uses a utility-based *selection* between one RL and one OED action.
    Its guarantee is only within the v10 local surrogate design objective,
    NOT a guarantee of lower true parameter error.
    """
    rng=rng or np.random.default_rng(seed+7777)
    details={}
    if policy=='random': return rng.uniform(-1,1,6).astype(np.float32),details
    if policy=='oed':
        a,u,parts=design_action(env,seed)
        return a,{'oed_predicted_utility':u,**parts}
    if policy in ('rl','hybrid'):
        if model is None:raise ValueError('Trained SAC model required for RL/hybrid policy')
        a_rl=np.asarray(model.predict(obs,deterministic=True)[0],dtype=np.float32)
        if policy=='rl':return a_rl,details
        a_oed,oed_u,_=design_action(env,seed)
        try: rl_u=predicted_action_utility(env,a_rl)
        except (RuntimeError,ValueError,np.linalg.LinAlgError,OverflowError):rl_u=-np.inf
        use_rl=bool(np.isfinite(rl_u) and rl_u>oed_u+hybrid_margin)
        details={'oed_predicted_utility':float(oed_u),'rl_predicted_utility':float(rl_u),
                 'used_rl':use_rl}
        return (a_rl if use_rl else a_oed),details
    if policy.startswith('tree'):
        if tree is None:raise ValueError('Trained distilled tree required')
        return np.clip(np.asarray(tree.predict(obs[None,:])[0],dtype=np.float32),-1,1),details
    raise ValueError(f'Unknown policy: {policy}')


def run_episode(env,policy,seed,model=None,tree=None,record_trace=False,
                hybrid_margin=0.):
    """Replay one complete three-nudge sequence on a newly seeded environment."""
    env.ecfg.record_trace=record_trace
    obs,_=env.reset(seed=int(seed))
    rng=np.random.default_rng(seed+7777)
    info_list=[];total=0.; actions=[];decisions=[]
    for nudge_index in range(3):
        a,decision=choose_action(policy,env,obs,seed,model,tree,rng,hybrid_margin)
        actions.append(np.asarray(a).tolist())
        decisions.append(decision)
        obs,reward,terminated,truncated,info=env.step(a)
        total+=float(reward);info_list.append(info)
        if terminated or truncated:break
    last=info_list[-1]
    err=last.get('debug_truth_errors',{})
    row={'seed':int(seed),'policy':policy,'success':(len(info_list)==3 and
        last.get('failure_reason') is None),
        'mass_error_kg':float(err.get('mass_error_kg',np.nan)),
        'com_error_mm':float(err.get('com_error_mm',np.nan)),
        'inertia_error_kgm2':float(err.get('inertia_error_kgm2',np.nan)),
        'reward':float(total),
        'estimated_sample_updates':int(last.get('estimator_updates',0)),
        'live_inertia_updates':int(last.get('continuous_inertia_previews',0)),
        'failure':last.get('failure_reason'),
        'hybrid_rl_selected':int(sum(d.get('used_rl',False) for d in decisions)),
    }
    # Ground truth is held only by the evaluator and is never included in the
    # Gymnasium observation/reward or planner. Component-wise diagnostic errors
    # are crucial: aggregate Frobenius error can obscure a poorly identified
    # product of inertia.
    if env.ecfg.debug_truth and env.truth_pi is not None and last.get('pi_est'):
        estimated=np.asarray(last['pi_est'],float)
        true_pi=np.asarray(env.truth_pi,float)
        for label,delta in zip(('m','mc_x','mc_y','mc_z','I_xx','I_yy','I_zz','I_xy','I_xz','I_yz'),
                               abs(estimated-true_pi)):
            row[f'abs_error_{label}']=float(delta)
    return row,{'seed':int(seed),'policy':policy,'actions':actions,
        'nudges':info_list,'decisions':decisions,'trace':env.estimation_trace if record_trace else []}


def evaluate(root,policies=('random','oed'),seeds=range(19,24),
             icfg=None,dcfg=None,model=None,tree=None,record_trace=False,
             sample_s=1.,randomized=True,progress=True,hybrid_margin=0.):
    icfg=icfg or IDConfig(sensor_sigma=1e-4)
    dcfg=dcfg or DesignConfig(candidate_count=64)
    rows=[];details=[]
    env=make_env(root,icfg,dcfg,record_trace=record_trace,
                 sample_s=sample_s,debug=True,randomized=randomized)
    try:
        for seed in seeds:
            for policy in policies:
                try:
                    row,detail=run_episode(env,policy,int(seed),model,tree,
                        record_trace=record_trace,hybrid_margin=hybrid_margin)
                    if not row['success']:
                        row.update({k:np.nan for k in ('mass_error_kg','com_error_mm','inertia_error_kgm2')})
                except (RuntimeError,ValueError,np.linalg.LinAlgError,OverflowError) as ex:
                    row={'seed':int(seed),'policy':policy,'success':False,
                         'mass_error_kg':np.nan,'com_error_mm':np.nan,
                         'inertia_error_kgm2':np.nan,'reward':np.nan,
                         'estimated_sample_updates':0,'live_inertia_updates':0,
                         'failure':str(ex),'hybrid_rl_selected':0}
                    detail={'seed':int(seed),'policy':policy,'error':str(ex)}
                rows.append(row);details.append(detail)
                if progress:print(f"seed={seed:>3} {policy:>6}: "
                    f"mass={row['mass_error_kg']:.4g} kg  "
                    f"CoM={row['com_error_mm']:.4g} mm  "
                    f"I={row['inertia_error_kgm2']:.4g} kg m²; "
                    f"success={row['success']}")
    finally:env.close()
    return pd.DataFrame(rows),details


def summarize(df):
    output=[]
    for policy,g in df.groupby('policy',sort=True):
        good=g[g.success]
        x={'policy':policy,'episodes':len(g),'successes':len(good)}
        for metric in ('mass_error_kg','com_error_mm','inertia_error_kgm2'):
            vals=good[metric].astype(float).dropna()
            for stat,fn in [('median',lambda x:np.median(x)),
                            ('mean',lambda x:np.mean(x)),
                            ('p90',lambda x:np.percentile(x,90))]:
                x[f'{metric}_{stat}']=float(fn(vals)) if len(vals) else np.nan
        x['mean_reward']=float(good.reward.mean()) if len(good) else np.nan
        output.append(x)
    return pd.DataFrame(output)


def per_parameter_summary(df):
    """Median absolute error of ALL TEN linear inertial parameters, per policy.

    m in kg, first moments mc in kg*m, origin tensor in kg*m^2.
    """
    columns=[f'abs_error_{name}' for name in
             ('m','mc_x','mc_y','mc_z','I_xx','I_yy','I_zz','I_xy','I_xz','I_yz')]
    good=df[df.success]
    available=[c for c in columns if c in good.columns]
    if not available:raise ValueError('Requires evaluate(..., debug_truth=True)')
    return good.groupby('policy')[available].median().T


def paired_wins(df,method_a='oed',method_b='rl',metric='inertia_error_kgm2'):
    pivot=df[df.success].pivot_table(index='seed',columns='policy',values=metric)
    if method_a not in pivot or method_b not in pivot:return None
    paired=pivot[[method_a,method_b]].dropna()
    if len(paired)==0:return None
    return {'metric':metric,'pairs':len(paired),
      'fraction_first_better':float(np.mean(paired[method_a]<paired[method_b])),
      'median_error_difference_first_minus_second':float(np.median(paired[method_a]-paired[method_b]))}


def save_evaluation(df,details,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    df.to_csv(folder/'episode_metrics.csv',index=False)
    summarize(df).to_csv(folder/'summary.csv',index=False)
    (folder/'rollouts.json').write_text(json.dumps(details,indent=2,default=str))
    return folder


def teacher_dataset(root,seed_list,teacher='oed',model=None,icfg=None,dcfg=None,
                    sample_s=1.,progress=True):
    """Independent episodes, OED teacher or deterministic RL; avoid truth leakage.

    Use **different** seed ranges from downstream evaluation and tuning.
    """
    icfg=icfg or IDConfig(sensor_sigma=1e-4)
    dcfg=dcfg or DesignConfig(candidate_count=64)
    env=make_env(root,icfg,dcfg,debug=False,sample_s=sample_s)
    X=[];Y=[];groups=[];utilities=[]
    try:
        for j,seed in enumerate(seed_list):
            o,_=env.reset(seed=int(seed))
            for k in range(3):
                action,diagnostics=choose_action(teacher,env,o,int(seed),model=model)
                X.append(o.copy());Y.append(np.asarray(action,dtype=np.float32))
                groups.append(int(seed))
                # The teacher action can be checked against its predicted utility.
                utilities.append(predicted_action_utility(env,action))
                o,r,done,trunc,info=env.step(action)
                if info.get('failure_reason'):raise RuntimeError(info['failure_reason'])
                if done or trunc:break
            if progress and (j+1)%max(1,len(seed_list)//5)==0:
                print(f'{j+1}/{len(seed_list)} teacher episodes')
    finally:env.close()
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(groups,int),np.asarray(utilities)


def behavior_clone_actor(model,observations,actions,epochs=80,batch_size=128,
                         learning_rate=3e-4,seed=19,verbose=True):
    """Imitation pretraining of SAC actor using OED teacher actions.

    Periodic angles are represented via cosine loss to respect the seam at -1/+1.
    This trains the actor *only*; SAC fine-tuning remains required.
    """
    import torch
    from torch.nn import functional as F
    observations=np.asarray(observations,np.float32);actions=np.asarray(actions,np.float32)
    if len(observations)<10:raise ValueError('Too few teacher decisions')
    rng=np.random.default_rng(seed)
    actor=model.policy.actor
    opt=torch.optim.Adam(actor.parameters(),lr=learning_rate)
    actor.train()
    hist=[]
    for ep in range(epochs):
        indexes=rng.permutation(len(observations))
        losses=[]
        for start in range(0,len(indexes),batch_size):
            idx=indexes[start:start+batch_size]
            o=torch.as_tensor(observations[idx],dtype=torch.float32,device=model.device)
            y=torch.as_tensor(actions[idx],dtype=torch.float32,device=model.device)
            predicted=actor(o,deterministic=True)
            # longitude, tangent angle are 2-periodic: -1 and +1 are equivalent.
            angle=1-torch.cos(np.pi*(predicted[:,[0,2]]-y[:,[0,2]]))
            other=F.smooth_l1_loss(predicted[:,[1,3,4,5]],y[:,[1,3,4,5]])
            loss=other+0.25*angle.mean()
            opt.zero_grad();loss.backward();opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append(float(np.mean(losses)))
    if verbose:print(f'Behavior cloning: {len(observations)} decisions, loss {hist[0]:.5f} -> {hist[-1]:.5f}')
    return hist


def distill_tree(X,Y,groups,depth_choices=(3,4,6,8,10),min_samples_leaf=4,
                 seed=19,output_dir=None):
    """Group-separated train/validation/test; tune depth on validation ONLY.

    Never split individual decisions from the same episode across partitions.
    Exports a compact piecewise-constant mathematical approximation of teacher.
    """
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.tree import DecisionTreeRegressor,export_text
    import joblib
    X=np.asarray(X,np.float32);Y=np.asarray(Y,np.float32);groups=np.asarray(groups)
    if X.shape[1]!=84 or Y.shape[1]!=6:raise ValueError('Expected 84 observation, 6 action dimensions')
    if len(np.unique(groups))<15:raise ValueError('Need at least 15 independent episodes')
    split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
    trainval,test=next(split.split(X,Y,groups))
    split2=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1)
    tri,vai=next(split2.split(X[trainval],Y[trainval],groups[trainval]))
    train=trainval[tri];valid=trainval[vai]
    trials=[]
    for depth in depth_choices:
        t=DecisionTreeRegressor(max_depth=int(depth),min_samples_leaf=min_samples_leaf,random_state=seed)
        t.fit(X[train],Y[train]);yhat=t.predict(X[valid])
        metric=float(np.mean(np.abs(yhat-Y[valid])))
        trials.append({'depth':int(depth),'validation_action_MAE':metric})
    best_depth=min(trials,key=lambda t:t['validation_action_MAE'])['depth']
    model=DecisionTreeRegressor(max_depth=best_depth,min_samples_leaf=min_samples_leaf,random_state=seed)
    model.fit(X[trainval],Y[trainval])
    test_pred=model.predict(X[test]);errors=np.abs(test_pred-Y[test])
    report={'teacher_samples':len(X),'unique_episodes':len(np.unique(groups)),
            'trainval_episodes':len(np.unique(groups[trainval])),
            'test_episodes':len(np.unique(groups[test])),
            'chosen_tree_depth':best_depth,
            'holdout_action_MAE':float(np.mean(errors)),
            'per_action_MAE':dict(zip(ACTION_NAMES,np.mean(errors,axis=0).astype(float).tolist())),
            'validation_trials':trials,
            'caution':'MAE is in [-1,1] normalized action units; periodic angles and trajectory effects need closed-loop evaluation.'}
    rules=export_text(model,feature_names=OBS_NAMES,max_depth=best_depth,decimals=4)
    if output_dir is not None:
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,output_dir/'interpretable_tree_v10.joblib')
        (output_dir/'policy_rules_v10.txt').write_text(
            'Decision tree is a symbolic APPROXIMATION, not a proven closed-form RL solution.\n\n'+
            json.dumps(report,indent=2)+'\n\n'+rules)
        (output_dir/'distillation_report_v10.json').write_text(json.dumps(report,indent=2))
    return model,report,rules


def plot_error_distributions(df):
    import matplotlib.pyplot as plt
    metrics=[('mass_error_kg','Mass (kg)'),('com_error_mm','CoM (mm)'),
             ('inertia_error_kgm2','Inertia Frobenius (kg·m²)')]
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    methods=list(df.policy.unique())
    for ax,(key,label) in zip(axes,metrics):
        data=[df.loc[(df.policy==p)&df.success,key].dropna().values for p in methods]
        ax.boxplot(data,tick_labels=methods,showfliers=True)
        ax.set_title(label);ax.set_ylabel('Absolute estimation error');ax.grid(alpha=.3)
    fig.tight_layout();return fig


def plot_continuous_trace(episode,truth_pi=None):
    """Time-resolved 10-parameter updates (not just post-nudge snapshots)."""
    import matplotlib.pyplot as plt
    from src.nudge_v10_math import inertia
    trace=episode['trace']
    if not trace:raise ValueError('Run evaluation with record_trace=True')
    arr=np.asarray(trace,float)
    time=arr[:,0]; est=arr[:,2:12]
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for i,label in enumerate(('mass kg','mc_x kg m','mc_y kg m','mc_z kg m')):
        ax=axes.flat[i]
        ax.plot(time,est[:,i],linewidth=1.2,label='estimate')
        if truth_pi is not None:ax.axhline(truth_pi[i],linestyle='--',label='truth (evaluation only)')
        ax.set_ylabel(label);ax.grid(alpha=.3)
        if i==0:ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Time (s)')
    fig.suptitle('Continuous mass and first-moment estimation');fig.tight_layout()
    fig2,axes2=plt.subplots(2,3,figsize=(13,6),sharex=True)
    labels=('Ixx','Iyy','Izz','Ixy','Ixz','Iyz')
    for i,ax in enumerate(axes2.flat):
        ax.plot(time,est[:,i+4],linewidth=1.1)
        if truth_pi is not None:ax.axhline(truth_pi[i+4],linestyle='--')
        ax.set_title(labels[i]);ax.grid(alpha=.3)
    for ax in axes2[-1]:ax.set_xlabel('Time (s)')
    fig2.suptitle('Continuous origin-frame inertia estimation (kg·m²)')
    fig2.tight_layout();return fig,fig2
