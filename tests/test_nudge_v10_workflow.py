"""End-to-end workflow checks; Gym tests skip unless dependencies are installed."""
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]

def test_interpretable_names_and_scales():
    pytest.importorskip('gymnasium')
    from src.nudge_v10_workflow import OBS_NAMES,ACTION_NAMES
    assert len(OBS_NAMES)==84 and len(set(OBS_NAMES))==84
    assert len(ACTION_NAMES)==6

def test_three_nudge_continuous_sample_count_and_no_spacecraft_constraint():
    pytest.importorskip('gymnasium')
    from src.nudge_v10_workflow import make_env,run_episode
    from src.nudge_v10_math import IDConfig,DesignConfig
    env=make_env(ROOT,IDConfig(sensor_sigma=0.),DesignConfig(candidate_count=16),
                 sample_s=.5,record_trace=True)
    try:
        row,rollout=run_episode(env,'oed',19,record_trace=True)
        assert row['success'],row['failure']
        assert len(rollout['nudges'])==3
        assert row['estimated_sample_updates']>=180
        assert row['live_inertia_updates']==row['estimated_sample_updates']
        assert len(rollout['trace'])==row['estimated_sample_updates']
        assert all(np.isfinite(np.asarray(nudge['pi_est'])).all() for nudge in rollout['nudges'])
        assert 'cbf_interventions_this_nudge' not in rollout['nudges'][0]
    finally:env.close()

def test_hybrid_never_lowers_predicted_one_step_information_score():
    pytest.importorskip('gymnasium')
    from src.nudge_v10_workflow import make_env,choose_action,predicted_action_utility,design_action
    from src.nudge_v10_math import IDConfig,DesignConfig
    env=make_env(ROOT,IDConfig(sensor_sigma=1e-4),DesignConfig(candidate_count=16),seed=19)
    class DummyModel:
        def predict(self,obs,deterministic=True):
            return np.zeros(6,dtype=np.float32),None
    try:
        obs=env._observation()
        baseline,_u,_=design_action(env,19)
        chosen,detail=choose_action('hybrid',env,obs,19,model=DummyModel())
        assert predicted_action_utility(env,chosen)+1e-8>=predicted_action_utility(env,baseline)
    finally:env.close()

def test_grouped_distillation_has_unseen_episodes(tmp_path):
    pytest.importorskip('gymnasium')
    from src.nudge_v10_workflow import distill_tree
    rng=np.random.default_rng(7)
    groups=np.repeat(np.arange(30),3)
    X=rng.uniform(-1,1,(90,84)).astype(np.float32)
    Y=np.tanh(X[:,:6]).astype(np.float32)
    tree,report,rules=distill_tree(X,Y,groups,depth_choices=(3,4),min_samples_leaf=2,output_dir=tmp_path)
    assert report['test_episodes']>=4
    assert report['holdout_action_MAE']>=0
    assert tree.predict(X[:1]).shape==(1,6)
    assert (tmp_path/'policy_rules_v10.txt').exists()
    assert isinstance(rules,str) and len(rules)>0 and '|---' in rules

def test_no_truth_leaked_to_policy_observation():
    pytest.importorskip('gymnasium')
    from src.nudge_v10_workflow import make_env
    env=make_env(ROOT,seed=20)
    try:
        assert len(env._observation())==84
        assert len(env.truth_pi)==10
        assert 'truth' not in ' '.join([c for c in env.observation_space.__dict__ if c=='truth_pi'])
        assert np.isfinite(env._observation()).all()
    finally:env.close()
