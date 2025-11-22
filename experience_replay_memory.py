#encoding=utf8
print('importing ExperienceReplayMemory...')
import random
import cv2
from log import log
from PIL import Image
import torchvision.transforms as transforms

class Transition(): 
    '''
    experience replay transition item
    '''
    # the function to transform state.image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    def __init__(self, state, action_id, reward, next_state, done): 
        '''
        init
        '''
        self.state      = state
        self.action_id  = action_id
        self.reward     = reward
        self.next_state = next_state
        self.done       = done

        # transform state.image and next_state.image
        if self.state.image_DQN is None: 
            log.info('transforms transition.state -> DQN')
            self.state.image_DQN = Transition.transform_state(self.state)

        if self.next_state.image_DQN is None: 
            log.info('transforms transition.next_state -> DQN')
            self.next_state.image_DQN = Transition.transform_state(self.next_state)


    @classmethod
    def transform_state(cls, state): 
        '''
        transform state:
            BGR -> RGB tensor
            to tensor
            no longer add a new axis
        '''
        image = cv2.cvtColor(state.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        pil_image = cls.transform(pil_image)
        # inputs = pil_image.unsqueeze(0)
        return pil_image


class ExperienceReplayMemory(): 
    '''
    experience replay memory D.
    '''

    def __init__(self): 
        '''
        initialize replay memory D to capacity N
        '''
        self.N                  = 6400
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
        if len(self.arr_transitions) < batch_size: 
            return None

        ret = random.sample(self.arr_transitions, batch_size)
        return ret

