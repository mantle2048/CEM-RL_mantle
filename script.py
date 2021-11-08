import os
import pickle
import json
import pandas as pd
import numpy as np

def gene_config(log_dir):

    for root, _, files in os.walk(log_dir):
        if 'parameters.txt' in files:
            print(root)
            config_name = os.path.join(root, 'config.json')
            parameters_name = os.path.join(root, 'parameters.txt')
            env_name = root.split('/')[-1].split('_')[-3]
            exp_name = '_'.join(['CEM-RL', env_name])
            with open(config_name, 'w') as  config, open(parameters_name) as  param:
                for line in param.readlines():
                    if line[0:4] == 'seed':
                        seed = int(line[-2])
                config_dict = dict(exp_name=exp_name, args=dict(seed=seed))
                json_data = json.dumps(config_dict)
                config.write(json_data)


def exchange_one_env(log_dir):

    max_len = 0
    for root, _, files in os.walk(log_dir):
        if 'log.pkl' in files:
            file_name = os.path.join(root, 'log.pkl')

            pkl_file = open(file_name, 'rb')

            data = pickle.load(pkl_file)
            step_len = len(data)
            max_len = max(max_len, step_len)

    for root, _, files in os.walk(log_dir):
        if 'log.pkl' in files:
            print(root, end=' ===>\n')

            file_name = os.path.join(root, 'log.pkl')
            out_name = os.path.join(root, 'progress.txt')

            pkl_file = open(file_name, 'rb')

            data = pickle.load(pkl_file)
            step_len = len(data)
            total_steps = np.linspace(25000, 1025000, max_len, endpoint=True, dtype=np.int64)

            data['total_steps'] = total_steps[:step_len]
            data.rename(columns = {"total_steps": "TotalEnvInteracts", "mu_score":"AverageTestEpRet"},  inplace=True)
            last_rew = data['AverageTestEpRet'].iloc[-1]
            data = data.append([{'TotalEnvInteracts':1100000,'AverageTestEpRet':last_rew}], ignore_index=True)
            data.to_csv(out_name, sep='\t',index=False)

def exchange_all_envs(envs_list):
    for env_dir in envs_list:
        exchange_one_env(env_dir)


if __name__ == '__main__':
    envs = [
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_AntBulletEnv-v0'),
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_Ant-v3'),
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_HalfCheetah-v3'),
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_Hopper-v3'),
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_Humanoid-v3')
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_Walker2d-v3'),
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_Walker2DBulletEnv-v0'),
        # os.path.join(os.getcwd(), 'tmp_CEM-RL', 'CEM-RL_Humanoid-v3')
        os.path.join(os.getcwd(), 'tmp_CEM-RL', 'Delay', 'F_CEM-RL_DelayHumanoid-v3')
    ]
    log_dir = os.path.join(os.getcwd(), 'tmp_CEM-RL')
    gene_config(log_dir)
    exchange_all_envs(envs)

