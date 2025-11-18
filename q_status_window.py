#encoding=utf8

import tkinter as tk
import time
from state_manager import State
from log import log
import numpy as np

class QStatus(): 
    '''
    Q status 
    '''
    def __init__(self): 
        '''
        init
        '''
        self.Q = None
        self.N = None

    def update(self, Q, N): 
        '''
        update
        '''
        self.Q = Q.copy()
        self.N = N.copy()


class QStatusWindow(): 
    '''
    Q status window
    '''

    def __init__(self, q_status, root): 
        '''
        init
        '''
        self.root = root

        w = 600
        h = 720

        x = 1280
        y = 0

        self.root.geometry("%dx%d+%d+%d" % (w, h, x, y))

        # frames
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, padx=10, pady=0)

        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=5)

        self.right_frame = tk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=5)

        # variables and labels
        self.variables  = {}
        self.labels     = {}

        self.add_label('title', self.top_frame)

        for i in range(0, 20): 
            if i in [9]: 
                continue

            key = 'state_%s' % (i)
            self.add_label(key, self.left_frame)

            key = 'Q_N_s_%s' % (i)
            self.add_label(key, self.right_frame)

        # data source
        self.q_status = q_status 


    def add_label(self, key, frame): 
        '''
        add a new label to the frame
        '''
        self.variables[key] = tk.StringVar()
        self.labels[key] = tk.Label(frame, textvariable=self.variables[key])
        self.labels[key].config(font=('Consolas', 12))
        if key == 'title': 
            self.labels[key].config(font=('Helvetica', 16))
        self.labels[key].pack(anchor="w", pady=5)


    def update(self): 
        '''
        use game_status to update local variables.
        then refresh UI.
        '''
        key = 'title'
        self.variables[key].set('%s' % ('Q & N'))
        self.labels[key].config(fg='blue')

        arr_state_name = [
            # 0-4
            'default',
            '突刺',
            '擒拿',
            '浮舟',
            '突刺*',

            # 5-8
            '受伤',
            'hulu',
            '追击',
            'crash',

            # 9 not exist
            '',

            # 10
            '主攻',

            # 11-19
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ]

        for i in range(0, 20): 
            key = 'state_%s' % (i)
            if key not in self.variables: 
                continue

            name = arr_state_name[i]
            self.variables[key].set('%s %s' % (i, name))

            state = State()
            state.final_state_id = i
            # state.action_space_key is wrong.

            key = 'Q_N_s_%s' % (i)
            if not self.q_status.Q.has(state): 
                line = '-'
                self.variables[key].set('%s' % line)
                continue

            arr_value = []
            arr_q_s = self.q_status.Q.get(state)
            for value in arr_q_s: 
                arr_value.append('%.2f' % (value))
            q_line = ', '.join(arr_value)
            n_line = '%s' % (self.q_status.N.get(state))
            line = q_line + ', N:' + n_line
            line = '[%d] ' % (np.argmax(arr_q_s)) + line
            self.variables[key].set('%s' % line)

        # refresh UI
        self.root.update_idletasks()
        self.root.update()


if __name__ == '__main__': 
    from storage import Storage
    obj_action_space = {
        'default': 5
    }
    Q = Storage(obj_action_space)
    N = Storage(obj_action_space)
    state = State()
    state.final_state_id = 0
    state.action_space_key = 'default'
    action_id = 0
    Q.set(state, action_id, 99.329823)
    N.set(state, action_id, 20)
    q_status = QStatus()
    q_status.Q = Q.copy()
    q_status.N = N.copy()
    root = tk.Tk()
    q_status_window = QStatusWindow(q_status, root)
    while True: 
        time.sleep(0.5)
        q_status_window.update()
