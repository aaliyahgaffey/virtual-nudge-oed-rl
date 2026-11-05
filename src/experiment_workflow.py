"""Notebook-first v11 planning, evaluation, offline coverage and RL comparisons."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.inertial_dynamics import IDConfig,DesignConfig,action_to_nudge,error_metrics,com_inertia
from src.nudge_environment_base import EnvConfig
from src.nudge_environment import InformationNudgeEnv
from src.information_gain import (InformationConfig,select_nudge,predicted_gain,
    component_summary, current_cov,theta_from_pi, NAMES)


def make_env(root,seed=None,noise=1e-4,candidates=32,sample_s=.5,debug=True,trace=False,info_config=None):
    env=InformationNudgeEnv(root,
        id_config=IDConfig(sensor_sigma=noise),
        design_config=DesignConfig(candidate_count=int(candidates)),
        env_config=EnvConfig(surrogate_sample_s=sample_s,debug_truth=debug,record_trace=trace),
        info_config=info_config or InformationConfig())
    if seed is not None:env.reset(seed=int(seed))
    return env


def choose_action(env,policy,obs,seed=0,model=None,tree=None):
    if policy=='random':
        return env._action_rng.uniform(-1.,1.,6).astype(np.float32), 'random'
    if policy=='oed_v11':
        s=select_nudge(env.surface,env.est,env.est.last_state,env.dcfg,env.icfg,
            seed=int(seed*1009+env.n*311),options=env.info_config)
        return s['action'],'oed_v11'
    if policy=='oed_v10':
        from src.nudge_optimization import optimize_hybrid_one
        a,_,_,_=optimize_hybrid_one(env.surface,env.est.last_state,env.est,
                                   env.dcfg,env.icfg,seed=int(seed*1009+env.n*311))
        return a,'oed_v10'
    if policy=='rl':
        if model is None:raise ValueError('RL model missing')
        a,_=model.predict(obs,deterministic=True)
        return np.asarray(a,np.float32),'rl'
    if policy=='tree':
        if tree is None:raise ValueError('Tree missing')
        return np.clip(tree.predict(np.asarray(obs)[None,:])[0],-1,1).astype(np.float32),'tree'
    if policy=='hybrid_v11':
        ao,_=choose_action(env,'oed_v11',obs,seed)
        ar,_=choose_action(env,'rl',obs,seed,model=model)
        scores=[]
        for a in (ao,ar):
            plan=action_to_nudge(a,env.surface,env.dcfg)
            u,*_=predicted_gain(env.est,env.est.last_state,plan,
                                env.dcfg,env.icfg,env.info_config)
            scores.append(u)
        if scores[1]>scores[0]:return ar,'rl'
        return ao,'oed_v11'
    raise ValueError(f'Unknown policy: {policy}')


def run_episode(env,seed,policy,model=None,tree=None,include_trace=False):
    obs,_=env.reset(seed=int(seed));records=[];selections=[];total=0.;done=False
    for k in range(3):
        action,source=choose_action(env,policy,obs,seed,model,tree)
        intended=action_to_nudge(action,env.surface,env.dcfg)
        pre_c,_=com_inertia(env.est.pi)
        body_angular_impulse=np.cross(intended.point-pre_c, intended.push*intended.impulse)
        obs,reward,done,truncated,info=env.step(action)
        total+=float(reward)
        selections.append(source)
        record={'seed':int(seed),'policy':policy,'nudge':int(k+1),
                'action':np.asarray(action,float).tolist(),'selected_from':source,
                'point_body_m':intended.point.tolist(),
                'push_body':intended.push.tolist(),
                'impulse_Ns':float(intended.impulse),
                'angular_impulse_about_est_CoM_Nms':body_angular_impulse.tolist(),
                'reward':float(reward),
                'estimator_updates':info.get('estimator_updates',0),
                'predicted_eig_nats':info.get('v11_predicted_eig_nats',np.nan),
                'passive_eig_nats':info.get('v11_passive_eig_nats',np.nan),
                'incremental_eig_over_passive_nats':info.get('v11_incremental_eig_nats',np.nan),
                'observed_covariance_proxy_nats':info.get('v11_observed_covariance_eig_proxy_nats',np.nan),
                'weakest_mode_before':info.get('v11_weak_modes_before'),
                'predicted_parts':info.get('v11_predicted_components'),
                'prequential_prediction_residual':info.get('prequential_prediction_residual'),
                'component_error_eval_only':info.get('v11_DEV_ONLY_coverage'),
                'failure_reason':info.get('failure_reason'),
                'pi_est':info.get('pi_est')}
        if env.ecfg.debug_truth and not info.get('failure_reason'):
            record.update(error_metrics(env.est.pi,env.truth_pi))
        records.append(record)
        if done or truncated:break
    success=done and len(records)==3 and not any(x['failure_reason'] for x in records)
    row={'seed':int(seed),'policy':policy,'success':bool(success),
         'total_reward':float(total),'selected_from_rl':selections.count('rl'),
         'continuous_estimator_updates':int(env.est.updates),
         'predicted_eig_nats_total':float(np.nansum([r['predicted_eig_nats'] for r in records])),
         'incremental_eig_over_passive_total':float(np.nansum([r['incremental_eig_over_passive_nats'] for r in records])),
         'observed_covariance_gain_proxy_total':float(np.nansum([r['observed_covariance_proxy_nats'] for r in records]))}
    if success and env.ecfg.debug_truth:
        row.update(error_metrics(env.est.pi,env.truth_pi))
        comp=component_summary(env.est.pi,env.truth_pi,current_cov(env.est))
        row.update({f'z_{name}':comp['z_score'][i] for i,name in enumerate(NAMES)})
        row.update({f'abs_err_{name}':comp['abs_error_physical'][i] for i,name in enumerate(NAMES)})
        row.update({f'coverage95_{name}':comp['within_95pct_normal'][i] for i,name in enumerate(NAMES)})
        row['NEES_proxy']=comp['quadratic_NES_approx']
    return row,records, (list(env.estimation_trace) if include_trace else [])


def evaluate(root,seeds,policies=('random','oed_v10','oed_v11','rl','hybrid_v11'),model=None,tree=None,
             noise=1e-4,candidates=32,sample_s=.5,progress=True):
    rows=[];nudges=[]
    env=make_env(root,noise=noise,candidates=candidates,sample_s=sample_s)
    try:
        for seed in seeds:
            for method in policies:
                if method in ('rl','hybrid_v11') and model is None:continue
                if method=='tree' and tree is None:continue
                try:
                    row,rs,_=run_episode(env,int(seed),method,model=model,tree=tree)
                    rows.append(row);nudges.extend(rs)
                    if progress: print(f'seed={seed} {method}: '+
                       ('mass={:.4f} kg; CoM={:.3f} mm; inertia={:.3f} kg m²'.format(
                           row['mass_error_kg'],row['com_error_mm'],row['inertia_error_kgm2'])
                        if row['success'] else 'FAILED'))
                except (ValueError,RuntimeError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as ex:
                    rows.append({'seed':int(seed),'policy':method,'success':False,'error':str(ex)})
                    if progress:print(f'seed={seed} {method}: {type(ex).__name__}: {ex}')
    finally:env.close()
    return pd.DataFrame(rows),nudges


def summary(df):
    metrics=['mass_error_kg','com_error_mm','inertia_error_kgm2',
             'total_reward','predicted_eig_nats_total','incremental_eig_over_passive_total','observed_covariance_gain_proxy_total']
    good=df[df.success]
    records=[]
    for method,g in df.groupby('policy'):
        s=good[good.policy==method]
        r={'policy':method,'episodes':len(g),'successes':len(s)}
        for key in metrics:
            if key not in s:continue
            values=s[key].dropna().astype(float)
            if len(values):
                r[key+'_median']=float(values.median())
                r[key+'_mean']=float(values.mean())
                r[key+'_p90']=float(values.quantile(.9))
        for name in NAMES:
            key=f'coverage95_{name}'
            if key in s:r[key+'_rate']=float(s[key].mean())
        records.append(r)
    return pd.DataFrame(records)


def paired_bootstrap(df,a='oed_v11',b='rl',metric='inertia_error_kgm2',draws=2000,seed=19):
    pivot=df[df.success].pivot_table(index='seed',columns='policy',values=metric)
    if a not in pivot or b not in pivot:return None
    p=pivot[[a,b]].dropna();delta=(p[a]-p[b]).values
    if len(delta)==0:return None
    rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(delta,(draws,len(delta)),replace=True),axis=1)
    return {'pairs':len(delta),'mean_paired_difference':float(delta.mean()),
            'median_paired_difference':float(np.median(delta)),
            'fraction_a_lower_error':float(np.mean(delta<0)),
            'mean_difference_bootstrap_95pct':np.quantile(means,[.025,.975]).tolist(),
            'note':'Bootstrap uncertainty reflects finite episode sampling, not model mismatch.'}


def offline_coverage(df,policy='oed_v11'):
    """Empirical coverage using simulation truth, for PRE-FLIGHT calibration only."""
    g=df[(df.policy==policy)&df.success]
    if len(g)==0:return pd.DataFrame()
    rows=[]
    for name in NAMES:
        v=np.asarray(g['z_'+name],float)
        rows.append({'parameter':name,'episodes':len(v),
                     'empirical_95pct_coverage':float(np.mean(np.abs(v)<=1.96)),
                     'median_abs_z':float(np.median(np.abs(v))),
                     'p90_abs_z':float(np.percentile(np.abs(v),90)),
                     'rms_z':float(np.sqrt(np.mean(v*v)))})
    return pd.DataFrame(rows)


def save(root,df,nudges,outdir):
    path=Path(root)/outdir
    path.mkdir(parents=True,exist_ok=True)
    df.to_csv(path/'final_episode_metrics.csv',index=False)
    summary(df).to_csv(path/'summary.csv',index=False)
    # Flatten log for direct comparisons, also retain nested detailed diagnostics.
    scalar=[{k:v for k,v in n.items() if isinstance(v,(int,float,str,bool)) or v is None} for n in nudges]
    pd.DataFrame(scalar).to_csv(path/'per_nudge_information.csv',index=False)
    (path/'per_nudge_details.json').write_text(json.dumps(nudges,indent=2,default=str))
    return path


def teacher_data(root,seeds,candidates=32,noise=1e-4,sample_s=.5,progress=False):
    env=make_env(root,noise=noise,candidates=candidates,sample_s=sample_s,debug=False)
    X=[];Y=[];groups=[]
    try:
        for seed in seeds:
            o,_=env.reset(seed=int(seed))
            for k in range(3):
                a,_=choose_action(env,'oed_v11',o,seed)
                X.append(o.copy());Y.append(a.copy());groups.append(seed)
                o,r,done,trunc,info=env.step(a)
                if info.get('failure_reason'):raise RuntimeError(info['failure_reason'])
                if done or trunc:break
    finally:env.close()
    return np.asarray(X,np.float32),np.asarray(Y,np.float32),np.asarray(groups,int)
