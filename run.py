import logging
import sys
import argparse

from config import set_path
from methods.mc_eval import mc_eval_sample
from methods.td_learning import td_zero_sample
from methods.ppo import ppo_sample
from methods.DQN import dqn_sample
from methods.sac import sac_sample
from ma_methods.IQL import iql_sample

def prepare_args():
    parser = argparse.ArgumentParser(
        description="the tool for running samples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mc_eval',
        action='store_true',
        help='running the mc evaluation sample in game: 21_points'
    )

    parser.add_argument(
        '--td_learning',
        type=int,
        help='running the TD learning sample in game: 21_points'
    )

    parser.add_argument(
        '--ppo',
        action='store_true',
        help='running the ppo'
    )

    parser.add_argument(
        '--dqn',
        action='store_true',
        help='running the dqn'
    )

    parser.add_argument(
        '--sac',
        action='store_true',
        help='running the sac'
    )

    parser.add_argument(
        '--iql',
        action='store_true',
        help='running the iql'
    )


    return parser.parse_args()

def main():
    logging.info("setting up project paths........")
    set_path()
    logging.info("paths set successfully.")

    args=prepare_args()

    if args.mc_eval:
        logging.info("starting mc evaluation sample........")
        mc_eval_sample()
    
    if args.td_learning==0:
        logging.info("starting td[0] learning sample........")
        td_zero_sample()
    
    if args.ppo:
        logging.info("starting PPO........")
        ppo_sample()
    
    if args.dqn:
        logging.info("starting DQN........")
        dqn_sample()
    
    if args.sac:
        logging.info("starting SAC........")
        sac_sample()
    
    if args.iql:
        logging.info("starting IQL........")
        iql_sample()

    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
