#encoding=utf8

print('importing...')
import random
from torchvision.models import resnet18, ResNet18_Weights
import torch

class Transition(): 
    '''
    experience replay transition item
    '''

    def __init__(self, state, action_id, reward, next_state, is_done): 
        '''
        init
        '''
        self.state      = state
        self.action_id  = action_id
        self.reward     = reward
        self.next_state = next_state
        self.is_done    = is_done


class ExperienceReplayMemory(): 
    '''
    experience replay memory D.
    '''

    def __init__(self): 
        '''
        # initialize replay memory D to capacity N
        '''
        self.N                  = 10000
        self.arr_transitions    = []


    def store(self, transition): 
        '''
        store the transition in D
        '''
        if len(self.arr_transitions) >= self.N: 
            self.arr_transitions.pop(0)

        self.arr_transitions.append(transition)


    def sample(self, batch_size): 
        '''
        sample random minibatch of transitions from D.
        '''
        ret = random.sample(self.arr_transitions, batch_size)
        return ret


class DQN(): 
    '''
    DQN
    '''

    def __init__(self, action_space): 
        '''
        init
        '''
        # learning rate
        self.lr = 0.01

        # every C steps, reset Q_hat = Q
        self.C      = 100
        self.step_i = 0


        self.action_space   = action_space
        self.num_classes    = action_space

        # create two networks
        # action-value function Q with weights theta
        # and target action-value function Q_hat with weights theta_bar = theta
        # I suppose it should be pronounced as theta bar.
        self.network        = create_network()
        self.target_network = create_network()

        '''
        '''
        '''
        '''
        '''
        '''
        # todo: check if the two networks has the same weight initially.
        '''
        '''
        '''
        '''
        '''
        '''
        self.target_network.load_state_dict(self.network.state_dict())

        # create optimizer using network's parameters.
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        # create loss function.
        self.loss_function = torch.nn.MSELoss()

        # the function to transform state.image
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


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


    def update_Q(arr_transition_batch): 
        '''
        update the network using experience replay batch
        '''
        # calculate y_j

        # if episode terminates at step j+1, y_j = r_j
        y = reward
        if not is_done: 
            Q_s_target = self.get_Q_target(state)
            # y_j = r_j + gamma * MAXa(Q_target(S j+1, a, theta_bar))
            y = reward + self.GAMMA * max_value_in_Q_s_next


        # performance a gradient descent step on (y_j - Q(S_j, a_j, theta)) ^ 2 with respect to the network parameters theta.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_i += 1

        # every C steps, reset Q_hat = Q
        if self.step_i % C == 0: 
            self.target_network.load_state_dict(self.network.state_dict())
    

    def get_Q(self, state): 
        '''
        get Q_s using network by state
        '''
        inputs = self.transform_state(state)

        with torch.no_grad(): 
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
            outputs = self.model(inputs)
            # _, predicted = torch.max(outputs, 1)
            # state.class_id = predicted.item()

            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            # todo: 看一下模型的output 输出值是什么
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''
            '''''''''''''''''''''''''''''''''''''''''

            return outputs


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
