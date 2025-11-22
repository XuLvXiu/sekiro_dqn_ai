#encoding=utf8

print('importing DQN...')
from torchvision.models import resnet18, ResNet18_Weights
from experience_replay_memory import Transition
import torch
from log import log
import time

class DQN(): 
    '''
    DQN (nature14236. 2015)
    '''

    def __init__(self, action_space): 
        '''
        init
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # learning rate
        self.LEARNING_RATE  = 0.01
        self.GAMMA = 0.85

        # every C steps, reset Q_hat = Q
        self.C      = 100
        self.step_i = 0

        self.num_classes    = action_space

        # create two networks
        # initialize action-value function Q with random weights theta
        # initialize target action-value function Q_hat with weights theta_bar = theta
        # I suppose it should be pronounced as theta bar.
        self.network        = self.create_network()
        self.target_network = self.create_network()

        # copy weights
        self.target_network.load_state_dict(self.network.state_dict())

        # check if the two networks have the same weights.
        for (name_1, p1), (name_2, p2) in zip(self.network.state_dict().items(),
                self.target_network.state_dict().items()): 
            if not name_1 == name_2: 
                print('two network state_dict() have different layer names', name_1, name_2)
                sys.exit(-1)

            if not torch.equal(p1, p2): 
                print('two network state_dict() have different params')
                sys.exit(-1)

        '''
        however, if we dump the two state_dict() to binary files, the MD5 of the files are different...
        I think it might be due to tensor floating point numbers.
        If you use torch.save, it will be even more weird.
        '''

        # create optimizer using network's parameters.
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.LEARNING_RATE)

        # create loss function.
        self.loss_function = torch.nn.MSELoss()

        # display the loss in the game status window
        self.loss = 0


    def create_network(self): 
        '''
        create the network
        '''
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_classes = self.num_classes
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        if torch.cuda.is_available(): 
            model = model.cuda()

        return model


    def dump(self): 
        '''
        dump the DQN
        '''
        obj = {
            'step_i': self.step_i,
            'network.state_dict': self.network.state_dict(),
            'target_network.state_dict': self.target_network.state_dict(),
            'optimizer.state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }
        log.info('DQN.dump, step_i:%s' % (self.step_i))
        return obj


    def load(self, obj): 
        '''
        load params and overwrite DQN
        '''
        self.step_i = obj['step_i']
        self.network.load_state_dict(obj['network.state_dict'])
        self.target_network.load_state_dict(obj['target_network.state_dict'])
        self.optimizer.load_state_dict(obj['optimizer.state_dict'])
        self.loss = obj['loss']
        log.info('DQN.load, step_i:%s' % (self.step_i))


    def update_Q(self, arr_transition_batch): 
        '''
        update the network using experience replay batch.
        we must use vector operations to speed up the process,
        since the game is still running...
        '''
        # convert list of object to seperated tensors of property.
        # 0.00555 s
        # can we make it tiny and faster?

        # t.state.image_DQN is tensor, so we can use torch.stack
        # [3, 224, 224]
        # print(arr_transition_batch[0].state.image_DQN.shape)
        arr_state       = torch.stack([t.state.image_DQN for t in arr_transition_batch])
        # [BATCH_SIZE, 3, 224, 224]
        # print(arr_state.shape)

        arr_action      = torch.tensor([t.action_id for t in arr_transition_batch], device=self.device)
        arr_reward      = torch.tensor([t.reward for t in arr_transition_batch], device=self.device)
        arr_next_state  = torch.stack([t.next_state.image_DQN for t in arr_transition_batch])
        arr_done        = torch.tensor([t.done for t in arr_transition_batch], device=self.device)


        # Q-learning (Watkins，1989)
        # Q(S, A) = Q(S, A) + alpha * (R + gamma * MAXa(Q(S', a)) - Q(S, A))

        # calculate y_j
        # 0.07569 s
        # y_j = r_j + gamma * MAXa(Q_hat(S j+1, a, theta_bar))
        self.target_network.eval()
        with torch.no_grad(): 
            inputs = arr_next_state
            if torch.cuda.is_available(): 
                inputs      = inputs.cuda()
            Q_s_next = self.target_network(inputs)
            max_value_in_Q_s_next, max_indices = Q_s_next.max(dim=1)
            y_j = arr_reward + self.GAMMA * max_value_in_Q_s_next

        # if episode terminates at step j+1, y_j = r_j
        '''
        weird bug: 
            RuntimeError: Index put requires the source and destination dtypes match, 
            got Float for the destination and Long for the source.
        I think we need to convert t.reward to float type.
        '''
        y_j[arr_done] = arr_reward[arr_done]

        max_log_items = 4
        log.debug('step_i[%s], Q_s_next[%s], max_value_in_Q_s_next[%s], reward[%s], done[%s], y_j[%s]' % (self.step_i, 
            Q_s_next[0:max_log_items], 
            max_value_in_Q_s_next[0:max_log_items], arr_reward[0:max_log_items], arr_done[0:max_log_items],
            y_j[0:max_log_items]))

        # calculate y
        # 0.07024 s
        self.network.train()
        inputs = arr_state
        if torch.cuda.is_available(): 
            inputs      = inputs.cuda()

        # torch.cuda.synchronize()
        # t1 = time.time()
        Q_s = self.network(inputs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(t2-t1)

        # from each row, get Q_s_a by action
        rows = torch.arange(Q_s.shape[0], device=self.device)
        Q_s_a = Q_s[rows, arr_action]
        y = Q_s_a

        log.debug('Q_s[%s], action[%s], Q_s_a|y[%s]' % (Q_s[0:max_log_items], 
            arr_action[0:max_log_items], y[0:max_log_items]))

        # performance a gradient descent step on (y_j - Q(S_j, a_j, theta)) ^ 2 
        # with respect to the network parameters theta.
        # 0.01423 s
        loss = self.loss_function(y, y_j)
        log.debug('loss[%s]' % (loss))
        self.loss = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_i += 1

        # every C steps, reset Q_hat = Q
        if self.step_i % self.C == 0: 
            log.info('reset Q_hat = Q after C[%s] steps' % (self.C))
            self.target_network.load_state_dict(self.network.state_dict())
    

    def get_Q(self, state): 
        '''
        get Q_s using network by state
        '''
        image   = Transition.transform_state(state)

        # add an axis
        inputs  = image.unsqueeze(0)

        self.network.eval()
        with torch.no_grad(): 
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
            outputs = self.network(inputs)
            return outputs.detach().cpu()


# test 
if __name__ == '__main__': 
    from experience_replay_memory import ExperienceReplayMemory, Transition
    from state_manager import State
    import cv2

    action_space = 3
    model = DQN(action_space)
    experience_replay_memory = ExperienceReplayMemory()
    BATCH_SIZE = 8

    arr_transition_batch = experience_replay_memory.sample(BATCH_SIZE)
    if arr_transition_batch is not None: 
        print('sample test error')
        sys.exit(0)

    # transition 1
    state = State()
    state.image = cv2.imread('./assets/20251027_084204_0.png')
    action_id = 1
    reward = 42.1
    next_state = State()
    next_state.image = cv2.imread('./assets/20251027_084204_45.png')
    done = False

    Q_s = model.get_Q(state)
    # Q_s tensor([[-0.0628, -0.2088,  0.8123]], device='cuda:0')
    print('Q_s', Q_s)

    t = Transition(state, action_id, reward, next_state, done)
    experience_replay_memory.store(t)

    arr_transition_batch = experience_replay_memory.sample(BATCH_SIZE)
    if arr_transition_batch is not None: 
        print('sample test error')
        sys.exit(0)

    # transition 2
    state = next_state
    action_id = 2
    reward = 100
    next_state = State()
    next_state.image = cv2.imread('./assets/20251027_084204_54.png')
    done = True

    t = Transition(state, action_id, reward, next_state, done)
    for i in range(0, BATCH_SIZE * 2): 
        experience_replay_memory.store(t)

    arr_transition_batch = experience_replay_memory.sample(BATCH_SIZE)
    if not len(arr_transition_batch) == BATCH_SIZE: 
        print('sample test error')
        sys.exit(0)

    t3 = time.time()
    model.update_Q(arr_transition_batch)
    t4 = time.time()
    # 0.17073 s
    print('update_Q total time: %.5f' % (t4 - t3))

