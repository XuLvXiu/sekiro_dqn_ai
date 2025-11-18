#encoding=utf8

'''
game environment
'''

print('importing env')
import grabscreen
import window
from window import BaseWindow, global_enemy_window, player_hp_window, boss_hp_window, player_posture_window
from utils import change_window
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from log import log
from actions import ActionExecutor
import cv2
import sys
import time
import signal
from pynput.keyboard import Listener, Key
import numpy as np
import os
from game_status_window import GameStatus, GameStatusWindow
from state_manager import State, StateManager
import shutil

class Env(object): 
    '''
    game env
    '''

    def __init__(self): 
        log.info('init env')
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.executor = ActionExecutor('./config/actions_conf.yaml')

        # actions in the game
        self.arr_action_name = ['IDLE', 'ATTACK', 'PARRY', 'SHIPO', 
            # 4
            'DOUBLE_ATTACK', 
            # 5
            'TRIPLE_ATTACK',
            # 6
            'QUADRUPLE_ATTACK',

            # 7
            'TAKE_HULU',
            # 8
            'STAND_UP',

            # 9
            'SHIPO_ATTACK',
            # 10
            'SHIPO_DOUBLE_ATTACK',
            # 11
            'SHIPO_TRIPLE_ATTACK',

            # 12
            'POSTURE_CRASH',
            'JUMP'
        ]

        self.IDLE_ACTION_ID                 = 0
        self.ATTACK_ACTION_ID               = 1
        self.PARRY_ACTION_ID                = 2
        self.SHIPO_ACTION_ID                = 3

        self.DOUBLE_ATTACK_ACTION_ID        = 4
        self.TRIPLE_ATTACK_ACTION_ID        = 5
        self.QUADRUPLE_ATTACK_ACTION_ID     = 6

        self.TAKE_HULU_ACTION_ID            = 7
        self.STAND_UP_ACTION_ID             = 8

        self.SHIPO_ATTACK_ACTION_ID         = 9
        self.SHIPO_DOUBLE_ATTACK_ACTION_ID  = 10
        self.SHIPO_TRIPLE_ATTACK_ACTION_ID  = 11
        self.POSTURE_CRASH_ACTION_ID        = 12

        # possible actions to be explored in RF
        # why we put it here?
        # it can be used both in train.py and main.py
        self.obj_possible_action_id = {
            'simple_attack': [
                self.ATTACK_ACTION_ID, 
                self.DOUBLE_ATTACK_ACTION_ID, 
                self.TRIPLE_ATTACK_ACTION_ID,
            ],
            'shipo_attack': [
                self.SHIPO_ATTACK_ACTION_ID,
                self.SHIPO_DOUBLE_ATTACK_ACTION_ID,
            ],
            'qinna_attack': [
                self.DOUBLE_ATTACK_ACTION_ID, 
            ],
            'fuzhou_attack': [
                self.ATTACK_ACTION_ID, 
                self.DOUBLE_ATTACK_ACTION_ID, 
            ],
            'parry': [
                self.PARRY_ACTION_ID,
            ],
            'player_hp_down': [
                self.STAND_UP_ACTION_ID,
            ],
            'player_posture_crash': [
                self.POSTURE_CRASH_ACTION_ID,
            ],
            'hulu': [
                self.TAKE_HULU_ACTION_ID,
            ],
            'player_posture_down': [
                self.ATTACK_ACTION_ID, 
                self.DOUBLE_ATTACK_ACTION_ID, 
            ],
            'parry_after_attack': [
                self.PARRY_ACTION_ID,
            ],
            'attack_after_damage': [
                self.DOUBLE_ATTACK_ACTION_ID,
            ],
            'default': [
                self.PARRY_ACTION_ID,
            ],
        }

        # state manager
        self.state_manager = StateManager()

        # training mode and eval mode have different HP threshold
        self.MODE_TRAIN = 'MODE_TRAIN'
        self.MODE_EVAL  = 'MODE_EVAL'
        self.mode = None

        self.previous_action_id = -1
        self.previous_player_hp = 100
        self.previous_boss_hp   = 100
        self.is_boss_dead = False
        self.is_player_dead = False
        self.player_life = 2

        # player posture
        self.previous_player_posture            = 0
        self.player_posture_high                = 0

        # the classification model v2
        self.model = None

        # Initialize camera
        grabscreen.init_camera(target_fps=12)
        # active and move window to top-left
        change_window.correction_window()

        if change_window.check_window_resolution_same(window.game_width, window.game_height) == False:
            raise ValueError(
                f"游戏分辨率和配置game_width({window.game_width}), game_height({window.game_height})不一致，请到window.py中修改"
            )

        if not self.wait_for_game_window_and_model(): 
            log.debug("Failed to detect game window.")
            raise ValueError('...')

        # game status window
        self.game_status = GameStatus()
        self.game_status_window = None

        # boss angry point, very naive
        self.boss_angry_point = 100


    def change_boss_angry_point(self, action, **kwargs): 
        '''
        change boss angry point according the action
        '''
        if action not in ['PLAYER_ATTACK', 'PLAYER_ATTACK_DAMAGE', 
                'TAKE_HULU', 'TIME_PASS', 'BOSS_ATTACK', 'BOSS_ATTACK_DAMAGE']: 
            print('action not found')
            sys.exit(-1)

        if action == 'PLAYER_ATTACK': 
            self.boss_angry_point += 20

        if action == 'PLAYER_ATTACK_DAMAGE': 
            attack_count = kwargs['attack_count']
            self.boss_angry_point += attack_count * 5

        if action == 'TAKE_HULU': 
            self.boss_angry_point += 30

        if action == 'TIME_PASS': 
            self.boss_angry_point -= 0.5

        if action == 'BOSS_ATTACK': 
            self.boss_angry_point -= 5

        if action == 'BOSS_ATTACK_DAMAGE': 
            self.boss_angry_point -= 6

        if self.boss_angry_point > 60: 
            self.boss_angry_point = 60 

        if self.boss_angry_point < 0: 
            self.boss_angry_point = 0


    def create_game_status_window(self): 
        '''
        create a tiny game status window if you like.
        '''
        self.game_status_window = GameStatusWindow(self.game_status)


    def update_game_status_window(self): 
        '''
        update game status window
        '''
        if self.game_status_window is not None: 
            self.game_status_window.update()


    def wait_for_game_window_and_model(self): 
        '''
        set window offset
        wait for model to load
        '''
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = 5
        print('num_classes:', num_classes)

        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

        model_file_name = 'model.resnet.v2'
        self.model.load_state_dict(torch.load(model_file_name))
        self.model.eval()

        if torch.cuda.is_available(): 
            self.model = self.model.cuda()

        while True: 
            frame = grabscreen.grab_screen()
            if frame is not None and window.set_windows_offset(frame):
                log.debug("Game window detected and offsets set!")

                BaseWindow.set_frame(frame)
                BaseWindow.update_all()

                log.debug('waiting for classifier model loading...')
                image = global_enemy_window.color.copy()

                state = State()
                state.image = image
                state.player_hp = 100
                state.boss_hp = 100
                state.is_player_hp_down = False
                state.is_boss_hp_down = False

                inputs = self.transform_state(state)

                with torch.no_grad(): 
                    if torch.cuda.is_available(): 
                        inputs = inputs.cuda()

                    outputs = self.model(inputs)

                return True
            time.sleep(1)

        return False


    def transform_state(self, state): 
        '''
        transform state:
            BGR -> RGB tensor
            add a new axis
        '''
        image = cv2.cvtColor(state.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        pil_image = self.eval_transform(pil_image)
        inputs = pil_image.unsqueeze(0)
        return inputs


    def on_action_finished(self):
        '''
        callback function of take_action
        '''
        log.debug("action execute finished")


    def is_parry(self, action_id): 
        '''
        check if action_id is parry related.
        '''
        if action_id == self.PARRY_ACTION_ID: 
            return True

        return False


    def is_attack(self, action_id): 
        '''
        check if action is attack
        '''
        if action_id == self.ATTACK_ACTION_ID: 
            return True

        if action_id == self.DOUBLE_ATTACK_ACTION_ID: 
            return True

        if action_id == self.TRIPLE_ATTACK_ACTION_ID: 
            return True

        if action_id == self.QUADRUPLE_ATTACK_ACTION_ID: 
            return True

        if action_id == self.SHIPO_ATTACK_ACTION_ID: 
            return True

        if action_id == self.SHIPO_DOUBLE_ATTACK_ACTION_ID: 
            return True

        if action_id == self.SHIPO_TRIPLE_ATTACK_ACTION_ID: 
            return True

        return False


    def is_take_hulu(self, action_id): 
        if action_id == self.TAKE_HULU_ACTION_ID: 
            return True

        return False


    def is_shipo(self, action_id): 
        if action_id == self.SHIPO_ACTION_ID: 
            return True

        return False


    def is_dianbu(self, action_id): 
        if action_id == self.SHIPO_ACTION_ID: 
            return True

        return False


    def take_action(self, action_id): 
        '''
        take a new action by action_id
        wait for the action to finish
        '''

        '''
        # no idle
        # if you want to idle, please parry.
        if action_id == self.IDLE_ACTION_ID: 
            action_id = self.PARRY_ACTION_ID
        '''

        # get action name
        action_name = self.arr_action_name[action_id]

        '''
        如果现在要防御, 那么需要判断前一个动作是否为防御，
            如果前一个动作也为防御，则 IDLE 即可，因为此时还未释放右键
        如果现在不要防御，那么也需要判断前一个动作是否为防御，
            如果前一个动作为防御，那么需要释放右键才能进行本次的动作。
        '''
        if self.is_parry(action_id) and self.executor.is_button_pressed('right'): 
            action_name = 'IDLE'

        if (not self.is_parry(action_id)) and self.executor.is_button_pressed('right'): 
            log.debug('take_action: %s' % ('RELEASE_PARRY'))
            self.executor.take_action('RELEASE_PARRY', action_finished_callback=self.on_action_finished)

        log.debug('take_action: %s' % (action_name))
        self.executor.take_action(action_name, action_finished_callback=self.on_action_finished)
        while self.executor.is_running(): 
            time.sleep(0.01)

        self.previous_action_id = action_id

        # wait for boss damage to take effect.
        # how about player damage taken?
        if self.is_attack(action_id): 
            # should check boss hp down or attack was blocked, to avoid sleep too long.

            # we attack the boss, he will be very angry.
            self.change_boss_angry_point('PLAYER_ATTACK')

            # since it is an attack, we will mark it.
            # it will be used by the following parry states.
            last_state = self.state_manager.get_last_state()
            last_state.is_attack = True
            last_state.is_parry_after_attack = False
            time.sleep(0.8)


    def check_done(self, state): 
        '''
        check if the player dead or if the boss dead(laugh).
        '''

        if self.mode is None: 
            log.error('please set env.mode to train() to eval()')
            sys.exit(-1)

        if self.mode == self.MODE_TRAIN: 
            if state.player_hp < 1: 
                self.is_player_dead = True
                self.player_life -= 1
                return True

            if state.boss_hp < 1: 
                self.is_boss_dead = True
                return True

        if self.mode == self.MODE_EVAL: 
            if state.player_hp < 1: 
                self.is_player_dead = True
                self.player_life -= 1
                return True

            if state.boss_hp < 1: 
                self.is_boss_dead = True
                return True

        return False


    def eval(self): 
        '''
        set mode
        '''
        self.mode = self.MODE_EVAL
        self.game_status.mode = 'EVAL'
        log.info('set new mode: %s' % (self.mode))


    def train(self): 
        '''
        set mode
        '''
        self.mode = self.MODE_TRAIN
        self.game_status.mode = 'TRAIN'
        log.info('set new mode: %s' % (self.mode))


    def reset(self): 
        '''
        reset the env
        '''
        self.previous_action_id = -1
        self.previous_player_hp = 100
        self.previous_boss_hp   = 100
        self.is_boss_dead       = False
        self.is_player_dead     = False
        self.player_life        = 2

        self.previous_player_posture            = 0
        self.player_posture_high                = 0
        self.boss_angry_point   = 100

        self.executor.interrupt_action()


    def stop(self): 
        '''
        terminate the env
        '''
        self.executor.interrupt_action()


    def get_state(self): 
        '''
        get a new state from env.
        calcaute player's hp and boss' hp
        '''
        log.debug('get new state begin')
        frame = grabscreen.grab_screen()
        BaseWindow.set_frame(frame)
        BaseWindow.update_all()

        # original image, BGR format
        image = global_enemy_window.color.copy()
        player_hp = player_hp_window.get_status()
        boss_hp = boss_hp_window.get_status()
        player_posture = player_posture_window.get_status()

        # should we move the codes below to state_manager.py?
        state = State()
        state.image = image
        state.player_hp = player_hp
        state.boss_hp = boss_hp
        state.is_player_hp_down = False
        state.is_boss_hp_down = False
        state.player_posture = player_posture

        player_hp_down = self.previous_player_hp - player_hp 
        boss_hp_down = self.previous_boss_hp - boss_hp
        THRESHOLD = 10
        if player_hp_down >= THRESHOLD: 
            state.is_player_hp_down = True
            state.is_player_hp_down_slightly = False

        if player_hp_down > 0 and player_hp_down < THRESHOLD: 
            state.is_player_hp_down = True
            state.is_player_hp_down_slightly = True

        THRESHOLD = 3
        if boss_hp_down > THRESHOLD: 
            state.is_boss_hp_down = True

        # check player posture crash
        if self.previous_player_posture > 50 and state.player_posture < 1: 
            state.is_player_posture_crash = True

        # if boss attacks player, its angry point should decrease a little.
        if player_posture > self.previous_player_posture: 
            self.change_boss_angry_point('BOSS_ATTACK')

        '''
        check posture down
        '''
        if player_posture < self.previous_player_posture: 
            # posture decrease to a reasonable value.
            state.is_player_posture_down_ok         = True

        if player_posture < 0.5: 
            # posture is close to 0
            state.is_player_posture_down_ok         = True

        # if boss is too angry, cancel the attack
        if self.boss_angry_point > 40: 
            state.is_player_posture_down_ok = False

        # if we are going to die, cancel the attack
        if player_hp < self.state_manager.HULU_THRESHOLD: 
            state.is_player_posture_down_ok = False

        # if our posture crashed, cancel the attack
        if state.is_player_posture_crash: 
            state.is_player_posture_down_ok = False

        # predict class id
        inputs = self.transform_state(state)

        with torch.no_grad(): 
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            state.class_id = predicted.item()

        # save it to state history manager.
        self.state_manager.save(state)

        # if we take hulu, boss will be a little angry.
        if state.state_id == self.state_manager.HULU_STATE_ID: 
            self.change_boss_angry_point('TAKE_HULU')

        # never modify the state from now on.

        log.debug('get new state end, hp: %5.2f %5.2f, class_id: %s, state_id: %s, arr_history_class_id: %s, posture: %.1f, is_attack: %s, is_parry_after_attack: %s, num_parry_steps_after_attack: %s, is_player_posture_down_ok: %s, action_space_key: %s' % (state.player_hp, 
            state.boss_hp, state.class_id, state.state_id, state.arr_history_class_id,
            state.player_posture, state.is_attack, state.is_parry_after_attack, state.num_parry_steps_after_attack,
            state.is_player_posture_down_ok, state.action_space_key))

        # update game status
        self.game_status.update_by_state(state)
        self.game_status.boss_angry_point = self.boss_angry_point
        self.update_game_status_window()

        return state


    def step(self, action_id): 
        '''
        take a step in the env
        get new state and calculate reward
        '''

        log.debug('new step begin, game action_id: %s' % (action_id))
        self.game_status.action_name = self.arr_action_name[action_id]
        self.update_game_status_window()

        self.take_action(action_id)

        new_state = self.get_state()

        is_done = self.check_done(new_state)
        (reward, log_reward) = self.cal_reward(new_state, action_id)

        # boss angry will descrease as time passes.
        self.change_boss_angry_point('TIME_PASS')

        log.debug('new step end, is_done[%s], is_dead[%s][%s], player_life[%s], reward[%s %s]' % (is_done, self.is_player_dead, self.is_boss_dead,
            self.player_life,
            reward, log_reward))

        return (new_state, reward, is_done)


    def cal_reward(self, new_state, action_id): 
        '''
        calculate the reward according to the action take and the new state

        打法: 立足防御，识别 boss 的 3个招式(1突刺危、2擒拿危擒擒又拿拿、3飞渡浮舟)，利用破绽进行尽可能多的连击，自身架势值不高的时候鼓励进攻
        '''
        reward = 10
        log_reward = '.'

        player_hp = new_state.player_hp
        boss_hp = new_state.boss_hp
        player_posture = new_state.player_posture

        log_reward += 'game action:%s,' % (action_id)
        
        if new_state.is_player_hp_down: 
            # the damage maybe caused by previous actions?
            reward -= 20
            if self.is_attack(action_id): 
                reward -= 30
                log_reward += 'is_attack,'

            log_reward += 'player_hp-,'

            # if boss attacks player, its angry point should decrease a little.
            self.change_boss_angry_point('BOSS_ATTACK_DAMAGE')

        if new_state.is_boss_hp_down: 
            reward += 30
            attack_count = int((self.previous_boss_hp - boss_hp) / 4)
            reward += (attack_count - 1) * 10
            log_reward += 'boss_hp-,attack_count:%s,' % (attack_count)

            # if we attacks boss multi times, he will be more angry.
            self.change_boss_angry_point('PLAYER_ATTACK_DAMAGE', attack_count=attack_count)

            if not self.is_attack(action_id): 
                # sometimes, the player is stunned when it pressed the left mouse button, 
                # after one or two steps, then the attack occurs.
                log.error('error: boss_hp-, but player is NOT attack')
                self.game_status.error = 'delayed boss_hp-'
                self.update_game_status_window()
            else: 
                if self.previous_player_posture < 50: 
                    reward += 10
                    log_reward += 'low_posture_effective_attack,'
        else: 
            if self.is_attack(action_id): 
                # even the boss-hp is not changed, player can interrupt boss-combo, or increase the posture of the boss.
                # however, attack will lose the opportunity to decrease player posture.
                if self.previous_player_posture < 50: 
                    reward += 10
                    log_reward += 'low_posture_ineffective_attack,'
                else: 
                    reward -= 5
                    log_reward += 'high_posture_ineffective_attack,'

                log_reward += 'boss_hp=,'

        if new_state.is_player_posture_crash: 
            reward -= 30
            log_reward += 'player_posture_crash'

        self.previous_player_hp = player_hp
        self.previous_boss_hp = boss_hp
        self.previous_player_posture = player_posture

        return (reward, log_reward)


    def go_to_next_episode(self): 
        '''
        close current episode, go to the next one.
        '''
        # wait for player death
        log.info('go_to_next_episode')
        time.sleep(60)
        log.info('confirm player death')
        log.debug('take_action: NEXT_EPISODE')
        self.executor.take_action('NEXT_EPISODE', action_finished_callback=self.on_action_finished)
        while self.executor.is_running(): 
            time.sleep(0.05)

    

if __name__ == '__main__': 
    is_debug = True
    global_is_running = False
    def signal_handler(sig, frame):
        log.debug("Gracefully exiting...")
        env.stop()
        sys.exit(0)

    def on_press(key):
        print('on_press: %s' % (key))
        global global_is_running
        try:
            if key == Key.backspace: 
                log.info('The user presses backspace in the game, will terminate.')
                env.stop()
                os._exit(0)

            if hasattr(key, 'char') and key.char == ']': 
                # switch the switch
                if global_is_running: 
                    global_is_running = False
                    env.stop()
                else: 
                    global_is_running = True

        except Exception as e:
            print(e)

    signal.signal(signal.SIGINT, signal_handler)
    keyboard_listener = Listener(on_press=on_press)
    keyboard_listener.start()

    def flush(): 
        if not is_debug: 
            return

        if len(arr_images) == 0: 
            return

        log.info('flush images: %s, do NOT power off.' % (len(arr_images)))
        image_dir = 'images/debug'
        if os.path.exists(image_dir): 
            shutil.rmtree(image_dir)
        if not os.path.exists(image_dir): 
            os.mkdir(image_dir)

        for item in arr_images[-1000:]: 
            cv2.imwrite(image_dir + '/' + item[1], item[0], [cv2.IMWRITE_PNG_COMPRESSION, 0])

        log.info('flush done')


    env = Env()
    # env.train()
    env.eval()
    env.reset()
    state = env.get_state()
    step_i = 0
    arr_old_class_id = [0, 0]
    arr_images = []
    previous_player_posture = -1
    player_posture_high = -1
    is_player_posture_high_happened = False
    while True: 
        log.info('main loop running, step_i: %s' % (step_i))
        if not global_is_running: 
            time.sleep(1.0)
            env.reset()
            state = env.get_state()
            step_i = 0
            flush()
            arr_images = []
            previous_player_posture = -1
            player_posture_high = -1
            is_player_posture_high_happened = False
            continue

        t1 = time.time()

        # get action from state
        inputs = env.transform_state(state)

        with torch.no_grad(): 
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
            outputs = env.model(inputs)
            _, predicted = torch.max(outputs, 1)
            action_id = predicted.item()

        class_id = action_id

        # if you modify this value, sync it to state_manager.py
        number_of_history_needed = 2
        if class_id == state.FUZHOU_CLASS_ID: 
            number_of_history_needed = 2

        signal_strength = 0
        for i in range(0, number_of_history_needed): 
            if state.arr_history_class_id[-1 * (i + 1)] == class_id: 
                signal_strength += 1

        log.debug('class_id: %s, signal_strength: %s, number_of_history_needed: %s' % (class_id, 
            signal_strength, number_of_history_needed))

        if signal_strength >= number_of_history_needed: 
            log.debug('signal_strength >= number_of_history_needed, class_id[%s]' % (class_id))
            if class_id == 0: 
                action_id = env.PARRY_ACTION_ID

            if class_id == 1: 
                action_id = env.SHIPO_ATTACK_ACTION_ID

            if class_id == 2: 
                action_id = env.QUADRUPLE_ATTACK_ACTION_ID

            if class_id == 3: 
                action_id = env.PARRY_ACTION_ID

            if class_id == 4: 
                # 4 is inaccurate
                action_id = env.PARRY_ACTION_ID
        else: 
            action_id = env.PARRY_ACTION_ID

        player_posture = player_posture_window.get_status()

        if player_posture > 25 and player_posture > previous_player_posture and player_posture < 75: 
            player_posture_high = player_posture
            is_player_posture_high_happened = True

        if action_id == env.PARRY_ACTION_ID and class_id == 0: 
            if player_posture < player_posture_high * 0.75 and is_player_posture_high_happened: 
                action_id = env.ATTACK_ACTION_ID
                is_player_posture_high_happened = False

        previous_player_posture = player_posture

        action_name = env.arr_action_name[action_id]
        log.info('class_id:%s, action_id:%s, action_name: %s' % (class_id, action_id, action_name))

        # save state for debug 
        arr_images.append((state.image.copy(), '%s_class_%s_action_%s_%s.png' % (step_i, class_id, action_id, action_name)))

        # do next step, get next state
        next_state, reward, is_done = env.step(action_id)
        t2 = time.time()
        log.info('main loop end one step, time: %.2f s' % (t2-t1))

        # prepare for next loop
        state = next_state

        step_i += 1
        arr_old_class_id.pop(0)
        arr_old_class_id.append(class_id)

        if is_done: 
            log.info('done.')
            break

    # end of while loop

    env.stop()

