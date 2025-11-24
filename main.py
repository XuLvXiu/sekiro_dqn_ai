#encoding=utf8

'''
main
'''
print('importing...')

import time
import sys
import signal
import cv2

from log import log
from pynput.keyboard import Listener, Key
import os
import numpy as np
from env import Env
import pickle
import json
from dqn import DQN
from experience_replay_memory import Transition, ExperienceReplayMemory
from rule import Rule
import torch

g_episode_is_running = False
def signal_handler(sig, frame):
    log.debug("Gracefully exiting...")
    env.stop()
    sys.exit(0)

def on_press(key):
    # print('on_press: %s' % (key))
    global g_episode_is_running
    try:
        if key == Key.backspace: 
            log.info('The user presses backspace in the game, will terminate.')
            env.stop()
            os._exit(0)

        if hasattr(key, 'char') and key.char == ']': 
            # switch the switch
            if g_episode_is_running: 
                g_episode_is_running = False
                env.stop()
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)


signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()

# create the DQN
action_space    = 2
obj_rule        = Rule()
model           = DQN(action_space)

# create game env
env = Env()
env.create_game_status_window()
env.eval()

# load Q
CHECKPOINT_FILE = 'checkpoint.pth'
JSON_FILE = 'checkpoint.json'

with open(JSON_FILE, 'r', encoding='utf-8') as f: 
    obj_information = json.load(f)

log.info(obj_information)
model.load(torch.load(CHECKPOINT_FILE))

env.game_status.episode = obj_information['episode']
env.game_status.DQN_loss    = model.loss
env.game_status.DQN_steps   = model.step_i

'''
# generate the policy
policy = {}
for (k, v) in Q.items(): 
    state = k
    Q_s = v
    action_id = np.argmax(Q_s)
    policy[state] = action_id
log.info('policy: %s' % (policy))
'''
arr_possible_action_id = env.arr_possible_action_id

env.reset()
state = env.get_state()

step_i = 0
while True: 
    # log.info('predict main loop running')
    if not g_episode_is_running: 
        print('if you lock the boss already, press ] to begin the episode')
        env.game_status.is_ai = False
        env.update_game_status_window()
        time.sleep(1.0)
        env.reset()
        state = env.get_state()
        continue

    env.game_status.is_ai = True

    t1 = time.time()

    env.game_status.step_i     = step_i
    env.game_status.error      = ''
    env.game_status.state_id   = state.final_state_id
    env.update_game_status_window()

    # get action by state from rule and Q
    action_id = obj_rule.apply(state, env)
    if action_id is None: 
        Q_s = model.get_Q(state)
        a_star = np.argmax(Q_s)
        log.debug('Q_s:%s, a_star: %s' % (Q_s, a_star))
        action_id = a_star

        # reduce the probability of attack.
        if action_id == 1: 
            if not state.is_player_posture_down_ok: 
                action_id = 0

    # at first, convert rl action_id to game action_id
    game_action_id = arr_possible_action_id[action_id]
    log.info('convert rl action_id[%s] to game action id[%s]' % (action_id, game_action_id))

    # do next step, get next state
    next_state, reward, is_done = env.step(game_action_id)
    t2 = time.time()
    log.info('predict main loop end one epoch, time: %.2f s' % (t2-t1))

    step_i += 1

    # prepare for next loop
    state = next_state

    if is_done: 
        log.info('is_done(hp < 1), re-check')
        # one more double attack to kill(REN_SHA, deathblow) the BOSS or take a new life.
        if env.is_boss_dead: 
            time.sleep(0.1)
            env.take_action(env.DOUBLE_ATTACK_ACTION_ID)
            # if boss dead, stop the game.

        if env.is_player_dead and env.player_life > 0: 
            time.sleep(10)
            env.take_action(env.ATTACK_ACTION_ID)
            time.sleep(1)
            env.is_player_dead = False

            # reset player previous posture
            # so the player will not consider its posture crashed after resurrection.
            env.previous_player_posture = 0

            state = env.get_state()
            # if take a new life, continue the game
            continue

        env.stop()
        log.info('done.')
        break

# end of while loop
env.update_game_status_window()

