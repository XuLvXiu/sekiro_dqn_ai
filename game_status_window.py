#encoding=utf8

import tkinter as tk
import time

class GameStatus(): 
    '''
    game status
    '''

    def __init__(self): 
        '''
        init
        '''
        self.player_hp          = 0
        self.boss_hp            = 0
        self.action_name        = '----'
        self.is_ai              = False
        self.state_id           = -1
        self.step_i             = 0
        self.episode            = 0
        self.error              = ''
        self.mode               = ''
        self.is_player_hp_down  = False
        self.is_boss_hp_down    = False
        self.player_posture     = 0

        self.DQN_loss           = 0
        self.DQN_steps          = 0


    def update_by_state(self, state): 
        '''
        use game state to update local variables.
        '''
        self.player_hp              = state.player_hp
        self.boss_hp                = state.boss_hp
        self.is_player_hp_down      = state.is_player_hp_down
        self.is_boss_hp_down        = state.is_boss_hp_down
        self.player_posture         = state.player_posture


class GameStatusWindow(): 
    '''
    a tiny tk window to show game status.
    '''

    def __init__(self, game_status): 
        '''
        init
        '''
        self.root = tk.Tk()
        self.root.title("Game Status")

        w = 300
        h = 500
        x = self.root.winfo_screenwidth() - w
        y = self.root.winfo_screenheight() - h - 100

        w = 600
        h = 300
        x = -1
        y = 720
        self.root.geometry("%dx%d+%d+%d" % (w, h, x, y))

        # frames
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, padx=10, pady=10)

        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # variables and labels
        self.variables  = {}
        self.labels     = {}

        self.add_label('is_ai', self.top_frame)

        self.add_label('state_id', self.left_frame)
        self.add_label('action_name', self.right_frame)

        self.add_label('player_hp', self.left_frame)
        self.add_label('boss_hp', self.right_frame)

        self.add_label('player_posture', self.left_frame)
        self.add_label('DQN_loss', self.right_frame)
        # self.add_label('empty', self.right_frame)

        self.add_label('episode', self.left_frame)
        self.add_label('DQN_steps', self.right_frame)

        # data source
        self.game_status = game_status


    def add_label(self, key, frame): 
        '''
        add a new label to the frame
        '''
        self.variables[key] = tk.StringVar()
        self.labels[key] = tk.Label(frame, textvariable=self.variables[key])
        self.labels[key].config(font=('Consolas', 16))
        if key == 'is_ai': 
            self.labels[key].config(font=('Helvetica', 48))
        self.labels[key].pack(anchor="w", pady=5)


    def update(self): 
        '''
        use game_status to update local variables.
        then refresh UI.
        '''
        key = 'is_ai'
        if self.game_status.is_ai: 
            self.variables[key].set('DQN AI')
            self.labels[key].config(fg='blue')
        else: 
            self.variables[key].set('人工')
            self.labels[key].config(fg='black')

        key = 'empty'
        if key in self.variables: 
            self.variables[key].set('')

        key = 'action_name'
        self.variables[key].set('%s' % (self.game_status.action_name))

        key = 'state_id'
        value = self.game_status.state_id
        if value == 10: 
            value = '10-DQN'
        self.variables[key].set('%s: %s' % (key, value))

        key = 'player_hp'
        self.variables[key].set('%s: %.2f' % (key, self.game_status.player_hp))
        self.labels[key].config(fg='black')
        if self.game_status.is_player_hp_down: 
            self.labels[key].config(fg='red')

        key = 'boss_hp'
        self.variables[key].set('%s: %.2f' % (key, self.game_status.boss_hp))
        self.labels[key].config(fg='black')
        if self.game_status.is_boss_hp_down: 
            self.labels[key].config(fg='red')

        key = 'player_posture'
        self.labels[key].config(fg='black')
        self.variables[key].set('%s: %.2f' % (key, self.game_status.player_posture))

        key = 'DQN_loss'
        self.labels[key].config(fg='black')
        self.variables[key].set('%s: %.4f' % (key, self.game_status.DQN_loss))

        key = 'episode'
        self.variables[key].set('[%s] %s: %s-%s' % (self.game_status.mode, 
            key, 
            self.game_status.episode, self.game_status.step_i))

        key = 'DQN_steps'
        self.variables[key].set('')
        self.labels[key].config(fg='black')
        self.variables[key].set('%s: %s' % (key, self.game_status.DQN_steps))

        # refresh UI
        self.root.update_idletasks()
        self.root.update()


if __name__ == '__main__': 
    game_status = GameStatus()
    game_status.is_ai = True
    window = GameStatusWindow(game_status)
    while True: 
        game_status.state_id += 1
        game_status.state_id %= 20
        time.sleep(0.5)
        window.update()
