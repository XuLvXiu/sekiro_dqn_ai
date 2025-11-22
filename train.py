#encoding=utf8

print('importing...')


from env import Env
import sys
import torch
from log import log
import cv2
import time
import signal
from pynput.keyboard import Listener, Key
import numpy as np
import os
import pickle
import json
import time
import argparse
from storage import Storage
from rule import Rule
from dqn import DQN
from experience_replay_memory import Transition, ExperienceReplayMemory

class Trainer: 
    '''
    train a DQN agent with rules and experience replay.
    the algorithm is in the file: DQNNaturePaper.pdf
    '''

    def __init__(self, is_resume=True): 
        '''
        init
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info('device: %s' % (self.device))

        self.env = Env()
        self.env.train()

        # possible actions to be explored in RL
        self.action_space = 3
        self.arr_possible_action_id = self.env.arr_possible_action_id[0:self.action_space]

        # episode parameters
        self.MAX_EPISODES = 1000
        self.next_episode = 0
        self.CHECKPOINT_FILE = 'checkpoint.pkl'
        self.JSON_FILE = 'checkpoint.json'

        log.info('is_resume: %s' % (is_resume))
        if is_resume: 
            obj_information = self.load_checkpoint()
            self.next_episode = obj_information['episode']

        # create game status window
        self.env.create_game_status_window()

        # create the DQN 
        self.DQN = DQN(self.action_space)

        # initialize replay memory D to capacity N
        self.experience_replay_memory = ExperienceReplayMemory()

        # small batch runs faster
        self.BATCH_SIZE = 64 / 2


    def train(self): 
        '''
        train
        '''
        begin_i = self.next_episode
        for i in range(begin_i, self.MAX_EPISODES): 
            # decay
            epsilon = 1.0 / ((i+1) / self.MAX_EPISODES + 1)
            log.info('episode: %s, epsilon: %s' % (i, epsilon))
            self.env.game_status.episode = i

            # one episode
            episode = self.generate_episode_from_Q_and_update_Q(epsilon)

            self.next_episode += 1

            obj_information = {
                'episode': self.next_episode,
                'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            }
            self.save_checkpoint(obj_information)

            self.env.go_to_next_episode()

        log.info('mission accomplished :)')
    

    def select_action_using_Q(self, state, epsilon): 
        '''
        select an action by state using Q
        '''
        action_space = self.action_space
        log.info('select_action_using_Q: using epsilon-greedy')

        # get Q_s using DQN's network by state
        Q_s = self.DQN.get_Q(state)

        # epsilon-greedy
        probs = self.get_probs(Q_s, epsilon, action_space)
        action_id = np.random.choice(action_space, p=probs)
        log.info('Q_s: %s, probs: %s, action_id: %s' % (Q_s, probs, action_id))
        return action_id


    def select_action(self, state, epsilon): 
        '''
        select an action by state using Q and rules.
        '''
        # rules: 
        obj_rule = Rule()
        action_id = obj_rule.apply(state, self.env)
        if action_id is not None: 
            log.info('select_action: state[%s], using predefined rule: [%s]' % (state.state_id, action_id))
            return action_id

        # Q: 
        action_id = self.select_action_using_Q(state, epsilon)
        return action_id


    def generate_episode_from_Q_and_update_Q(self, epsilon): 
        '''
        generate an episode using epsilon-greedy policy
        '''
        env = self.env

        # init S
        env.reset()
        state = env.get_state()

        step_i = 0

        while True: 
            # log.info('generate_episode main loop running')
            if not g_episode_is_running: 
                print('if you lock the boss already, press ] to begin the episode')
                self.env.game_status.is_ai = False
                self.env.update_game_status_window()
                time.sleep(1.0)

                # init S
                env.reset()
                state = env.get_state()

                step_i = 0
                continue


            self.env.game_status.is_ai = True

            t1 = time.time()
            log.info('generate_episode step_i: %s,' % (step_i))

            # select action(A) by state(S) using rules and Q
            action_id = self.select_action(state, epsilon)

            self.env.game_status.step_i     = step_i
            self.env.game_status.error      = ''
            self.env.game_status.state_id   = state.final_state_id
            self.env.update_game_status_window()

            # take action(A), get reward(R) and next state(S')
            # at first, convert RL action_id to game action_id
            game_action_id = self.arr_possible_action_id[action_id]
            log.info('convert rl action_id[%s] to game action id[%s]' % (action_id, game_action_id))
            next_state, reward, is_done = env.step(game_action_id)

            # if current state is a DQN state, we need to do more work on DQN.
            if state.state_id == self.env.state_manager.DQN_STATE_ID: 
                t3 = time.time()
                # if next state is not a DQN state, this means the episode is done for DQN
                done = is_done
                if not next_state.state_id == self.env.state_manager.DQN_STATE_ID: 
                    done = True

                # store the transition(S_t, a_t, r_t, S_t+1) in D
                transition = Transition(state, action_id, reward, next_state, done)
                self.experience_replay_memory.store(transition)

                # sample random minibatch of transitions from D
                arr_transition_batch = self.experience_replay_memory.sample(self.BATCH_SIZE)

                if arr_transition_batch is not None: 
                    if not len(arr_transition_batch) == self.BATCH_SIZE: 
                        print('something is wrong with the experience_replay_memory.sample')
                        sys.exit(-1)
                    # update Q(aka: network)
                    self.DQN.update_Q(arr_transition_batch)
                t4 = time.time()
                log.info('DQN total time: %.2f s' % (t4-t3))

            # everything is done.
            # prepare for next step
            # S = S'
            state = next_state

            t2 = time.time()
            log.info('generate_episode main loop end one step, time: %.2f s' % (t2-t1))
            step_i += 1

            if is_done: 
                env.stop()
                log.info('done.')
                break

        # end of while loop

        log.info('episode done. length: %s' % (step_i))
        self.env.update_game_status_window()


    def get_probs(self, Q_s, epsilon, action_space): 
        '''
        obtain the action probabilities related to the epsilon-greedy policy.
        '''
        ones = np.ones(action_space)
        # default action probability
        policy_s = ones * epsilon / action_space

        # best action probability
        a_star = np.argmax(Q_s)
        log.info('a_star: %s' % (a_star))
        policy_s[a_star] = 1 - epsilon + epsilon / action_space

        return policy_s


    def save_checkpoint(self, obj_information): 
        '''
        save checkpoint for future use.
        '''
        log.info('save_checkpoint...')
        log.info('do NOT terminate the power, still saving...')
        # log.info('actions: %s' % (self.env.arr_action_name))
        
        log.info('still saving...')

        # write json information
        log.debug(obj_information)
        with open(self.JSON_FILE, 'w', encoding='utf-8') as f: 
            json.dump(obj_information, f, indent=4, ensure_ascii=False) 

        log.info('saved ok')


    def load_checkpoint(self): 
        '''
        load history checkpoint
        '''
        log.info('load_checkpoint')
        obj_information = {'episode': 0}
        try: 
            with open(self.JSON_FILE, 'r', encoding='utf-8') as f: 
                obj_information = json.load(f)
        except Exception as e: 
            log.error('ERROR load checkpoint: %s', (e))

        log.debug(obj_information)
        return obj_information


    def stop(self): 
        '''
        stop the trainer
        '''
        self.env.stop()



# main
parser = argparse.ArgumentParser()
parser.add_argument('--new', action='store_true', help='new training', default=False)
args = parser.parse_args()
is_resume = not args.new

g_episode_is_running = False
def signal_handler(sig, frame):
    log.debug("Gracefully exiting...")
    t.stop()
    sys.exit(0)

def on_press(key):
    # print('on_press: %s' % (key))
    global g_episode_is_running
    try:
        if key == Key.backspace: 
            log.info('The user presses backspace in the game, will terminate.')
            t.stop()
            os._exit(0)

        if hasattr(key, 'char') and key.char == ']': 
            # switch the switch
            if g_episode_is_running: 
                # g_episode_is_running = False
                # t.stop()
                print('I cannot stop myself lalala')
            else: 
                g_episode_is_running = True

    except Exception as e:
        print(e)

signal.signal(signal.SIGINT, signal_handler)
keyboard_listener = Listener(on_press=on_press)
keyboard_listener.start()

t = Trainer(is_resume)
t.train()
