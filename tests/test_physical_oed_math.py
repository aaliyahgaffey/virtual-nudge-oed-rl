"""Pure-NumPy/SciPy tests; no MuJoCo package needed."""
import unittest
import numpy as np
from src.physical_oed_identification import (
 OriginState,momentum_regressor,skew,matrix_inertia,com_and_ic,
 SpatialIdentifier,segment_rows,svd_observability,e_opt_score,
 predict_post,normalize)

class OEDMathTests(unittest.TestCase):
    def setUp(self):
        self.pi=np.array([620.,20.,-12.,8.,70.,76.,82.,1.5,-2.,.5])
        self.s=OriginState(r=np.array([3.,2.,-1.]),v=np.array([.2,-.03,.05]),
          R=np.eye(3),omega_body=np.array([.013,-.025,.031]))

    def test_skew_cross(self):
        np.testing.assert_allclose(skew(self.s.v)@self.s.r,np.cross(self.s.v,self.s.r))
    def test_momentum_linear_part(self):
        c,Ic=com_and_ic(self.pi)
        P=momentum_regressor(self.s)@self.pi
        expected=self.pi[0]*(self.s.v+np.cross(self.s.omega_body,c))
        np.testing.assert_allclose(P[:3],expected)
    def test_momentum_angular_part(self):
        c,Ic=com_and_ic(self.pi)
        P=momentum_regressor(self.s)@self.pi
        h=np.cross(self.s.r,P[:3])+matrix_inertia(self.pi)@self.s.omega_body
        h+=np.cross(self.pi[1:4],self.s.v)
        np.testing.assert_allclose(P[3:],h)
    def test_origin_to_com_parallel_axis(self):
        c,Ic=com_and_ic(self.pi)
        np.testing.assert_allclose(Ic,matrix_inertia(self.pi)-self.pi[0]*
                                  ((c@c)*np.eye(3)-np.outer(c,c)))
    def test_svd_projector_identity(self):
        A=np.eye(10)
        a=svd_observability(A)
        np.testing.assert_allclose(a['P_observable'],np.eye(10))
        np.testing.assert_allclose(a['P_unobservable'],np.zeros((10,10)),atol=1e-12)
    def test_svd_nullspace(self):
        A=np.zeros((12,10));A[:6,:6]=np.eye(6)
        a=svd_observability(A)
        self.assertEqual(a['rank'],6)
        np.testing.assert_allclose(a['P_unobservable']@a['P_unobservable'],
                                   a['P_unobservable'],atol=1e-12)
    def test_consecutive_zero_forces(self):
        A,b=segment_rows(self.s,[self.s,self.s],np.zeros(6))
        np.testing.assert_allclose(A,np.zeros((12,10)))
        np.testing.assert_allclose(b,np.zeros(12))
    def test_e_gain_monotone(self):
        score,parts=e_opt_score(np.eye(10),np.eye(10)*2.)
        self.assertGreater(score,0)
        self.assertAlmostEqual(parts['E_gain'],2.)
    def test_synthetic_post_is_finite(self):
        point=np.array([.45,0.,0.])
        j=np.array([-1.,0.,0.])
        z=predict_post(self.s,self.pi,point,j,1.,np.array([.001,2.,4.]))
        self.assertEqual(len(z),3)
        for t in z:self.assertTrue(np.isfinite(momentum_regressor(t)).all())
        impulse=np.r_[j,np.cross(self.s.r+point,j)]
        delta=(momentum_regressor(z[0])-momentum_regressor(self.s))@self.pi
        np.testing.assert_allclose(delta,impulse,rtol=0,atol=1e-8)
    def test_estimator_initial_prior_positive(self):
        estimator=SpatialIdentifier()
        c,Ic=com_and_ic(estimator.pi)
        self.assertGreater(np.linalg.eigvalsh(Ic)[0],0.)

if __name__=='__main__':unittest.main()
