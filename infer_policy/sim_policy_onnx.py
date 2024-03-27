import numpy as np
import torch
import onnxruntime as ort
import onnx
import torch.nn.functional as F
from utils.wrappers import NormalizedBoxEnv
from JYLite_env_meta.envs.gym_envs.JYLite_gym_env import JYLiteGymEnv


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = np.minimum(np.maximum(sigmas_squared,1e-7),np.max(sigmas_squared))
    sigma_squared = 1. / np.sum(np.reciprocal(sigmas_squared),axis=0)
    mu = sigma_squared * np.sum(mus / sigmas_squared,axis=0)
    return mu, sigma_squared


class SimPolicy():
    def __init__(self):
        '''self.use_ib = False'''
        self.context_encoder_path = 'onnx_model/' + 'meta_context_encoder_ref_meta_rl_model.onnx'
        self.policy_path = 'onnx_model/' + 'meta_policy_ref_meta_rl_model.onnx'
        self.context_encoder = onnx.load(self.context_encoder_path)
        onnx.checker.check_model(self.context_encoder)
        self.policy = onnx.load(self.policy_path)
        onnx.checker.check_model(self.policy)
        self.context_sess = ort.InferenceSession(self.context_encoder_path,providers=['CPUExecutionProvider'])
        self.policy_sess = ort.InferenceSession(self.policy_path,providers=['CPUExecutionProvider'])
        self.context_input_name = self.context_sess.get_inputs()[0].name
        self.context_output_name = self.context_sess.get_outputs()[0].name
        self.policy_input_name = self.policy_sess.get_inputs()[0].name
        self.policy_output_name = self.policy_sess.get_outputs()[0].name
        self.env = JYLiteGymEnv(render=True,gait="straight",enable_disabled=True)
        self.env = NormalizedBoxEnv(self.env)
        self.max_path_length = 200
        self.latent_dim = 5
        self.z_means = np.zeros((1,self.latent_dim)).astype(np.float32)
        self.z_vars = np.ones((1,self.latent_dim)).astype(np.float32)
        self.use_ib = True
        self.use_next_obs_in_context = False
        self.sample_z()
        self.context = None

    def sample_z(self):
        if self.use_ib:
            posteriors = np.random.normal(loc=0.0,scale=1.0,size=(1,self.latent_dim)).astype(np.float32)
            self.z = posteriors
        else:
            self.z = self.z_means
            print("The self.z is:",self.z)

    def get_action(self,obs):
        z = self.z  # [1,5]
        obs = np.expand_dims(obs,axis=0).astype(np.float32) # [1,37]
        in_ = np.concatenate((obs,z),axis=1)                # [1,42]
        action = self.policy_sess.run([self.policy_output_name],{self.policy_input_name:in_})[0][0]
        return action


    def infer_posterior(self,context): # context: [200,50]
        params = self.context_sess.run([self.context_output_name],{self.context_input_name:context})[0]  # [200,10]
        params = np.array(params)
        if self.use_ib:
            mu = params[...,:self.latent_dim]  # [200,5]
            sigma_squared = params[...,self.latent_dim:]
            sigma_squared = F.softplus(torch.from_numpy(sigma_squared)).numpy()
            z_params = [_product_of_gaussians(m,s) for m,s in zip(mu,sigma_squared)]
            self.z_means = np.stack([p[0] for p in z_params])
            self.z_vars = np.stack([p[1] for p in z_params])
        else:
            self.z_means = np.expand_dims(np.mean(params,axis=0),axis=0)  # [1,5]
        self.sample_z()


    def update_context(self,inputs):
        o,a,r,no,d,info = inputs
        o = o[None,...].astype(np.float32)
        a = a[None,...].astype(np.float32)
        r = np.array([r])[None,...].astype(np.float32)
        no = no[None,...].astype(np.float32)

        if self.use_next_obs_in_context:
            data = np.concatenate((o,a,r,no),axis=1)
        else:
            data = np.concatenate((o,a,r),axis=1)
        if self.context is None:
            self.context = data
        else:
            self.context = np.concatenate([self.context,data],axis=0)  # [now_path_length,50]


    def rollout(self,accum_context=True):
        observations = []
        actions = []
        rewards = []
        terminals = []
        env_infos = []
        o,info = self.env.reset()
        next_o = None
        path_length = 0
        while path_length < self.max_path_length:
            a = self.get_action(o)
            next_o,r,d,env_info = self.env.step(a)
            if accum_context:
                self.update_context([o,a,r,next_o,d,env_info])
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            path_length += 1
            o = next_o
            env_infos.append(env_info)
            if d:
                break
        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions,1)
        observations = np.array(observations)
        if len(observations.shape) == 1:
            observations = np.expand_dims(observations,1)
            next_o = np.array([next_o])
        next_observations = np.vstack((observations[1:,:],np.expand_dims(next_o,0)))
        return dict(
            observations = observations,
            actions = actions,
            rewards = np.array(rewards).reshape(-1,1),
            next_observations = next_observations,
            terminals = np.array(terminals).reshape(-1,1),
            env_infos = env_infos)



if __name__ == "__main__":
    sim = SimPolicy()
    sim.env.reset_task(idx=4,specify_pos=np.array([-1.2,1.8]))
    path_1 = sim.rollout(accum_context=True)
    sim.infer_posterior(sim.context)
    path_2 = sim.rollout(accum_context=False)





















