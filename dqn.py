import warnings
warnings.filterwarnings("ignore")
import math,time,random,getopt,sys,os
from collections import namedtuple, deque
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import torchvision.transforms as T
import gym
from gym import ObservationWrapper
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.monitoring.video_recorder import VideoRecorder



##### CONSTANTS #####
STACKING = 6 # Number of frames to stack together (set to 1 for no stacking)
SEED = 1 # To control randomness (DO NOT MODIFY)
ENV_NAME = "Mineline-v0"
# FOR TRAINING
NB_EPISODES = 1000 # Number of episodes for training
RENDER_MODE_TRAIN = False # Set to True to render simulation during training
# FOR TESTING
RENDER_MODE_TEST = False # Set to True to render simulation during testing
RECORD_ALL = True # True to record all tests, False to record last test
# Hyper parameters 
MEMORY_SIZE = 50000 # Size of the Replay Memory of the agent
EXPLORATION_RATE = 0.99 # Randomness of the agent's action 
EXPLORATION_DECAY_RATE = 0.9998 # To reduce exploration rate with exploration decay method
MIN_EXPLORATION_RATE = 0.05 # Minimum exploration rate
BATCH_SIZE = 32 # DNN training batch size
LEARNING_RATE = 0.0003 # DNN optimizer learning rate
GAMMA = 0.99 # DQN discount factor
TARGET_NN_UPDATE = 10000 ##NOT USED## # Update target network after X learned examples (method 1)
TARGET_NN_UR = 0.01 # Update target network with an update rate (method 2)

# To load tensors on GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class CustomObservation(ObservationWrapper):
    '''
    Observation wrapper that modify the observation to get POV Box.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space['pov']

    def observation(self, observation):
        return np.array(observation['pov'])



class ReplayMemory(object):
    '''
    Circular memory of a DQN.
    '''
    def __init__(self, size):
        self.memory = deque([],size) #Deque automatically remove eldest experiences

    def push(self, *args):
        '''Save an experience in the memory'''
        self.memory.append(ReplayMemory.Template()(*args))

    def sample(self, batch_size):
        '''Take a random sample of given size'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def Template():
        '''Return data template stored in the memory'''
        return namedtuple('Transition',('state', 'action', 'next_state', 'reward'))



class DQN_CNN(nn.Module):
    '''
    Convolutional neural network class.
    '''
    # input_channels is specified for frame stacking
    # action_space is output size
    def __init__(self, input_channels, action_space):
        super().__init__()
        
        # Network artchitecture
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
            nn.Flatten(),
            nn.Linear(7*7*32,128),
            nn.ReLU(),
            nn.Linear(128,action_space)
        )
    
    def forward(self, x):
        '''Forward propagation'''
        return self.network(x)



class DQNAgent():
    '''
    Deep Q-Learning Agent.
    '''
    def __init__(self, state_space, action_space, input_channels, 
            memory_size=MEMORY_SIZE, 
            exploration_rate=EXPLORATION_RATE,
            decrease_rate=EXPLORATION_DECAY_RATE,
            min_exploration_rate=MIN_EXPLORATION_RATE,
            training_batch_size=BATCH_SIZE,
            model_learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            target_nn_update=TARGET_NN_UPDATE,
            target_nn_update_rate=TARGET_NN_UR,
            is_training = True,
            device=torch.device("cpu")):
        
        # Device to use (GPU or CPU)
        self.device = device

        # Network input and output size
        self.state_space = state_space
        self.action_space = action_space
        
        # Model and Target Networks
        self.is_training = is_training
        self.model = DQN_CNN(input_channels, action_space).to(device)
        self.target_model = DQN_CNN(input_channels, action_space).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval() # Target model is used in evaluation mode
        # Hyper-parameters
        self.batch_size = training_batch_size
        self.gamma = gamma
        # Optimizer
        self.criterion = nn.HuberLoss() #nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_learning_rate)
        # To update target network
        self.nb_train = 0
        self.target_update = target_nn_update
        self.target_ur = target_nn_update_rate
        
        # Replay memory
        self.memory = ReplayMemory(memory_size)

        # Exploration
        self.eps = exploration_rate
        self.dcr = decrease_rate
        self.min_eps = min_exploration_rate


    def save_model(self, path='model.pth', with_optimizer=False, epoch=None):
        '''Save DNN model to a file'''
        if with_optimizer: # To keep training later
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, path)
        else:
            torch.save({
                'model': self.model.state_dict()
            }, path)


    def load_model(self, path='model.pth', with_optimizer=False):
        '''Load DNN model from a file'''
        save = torch.load(path)
        self.model.load_state_dict(save['model'])
        self.target_model.load_state_dict(save['model'])
        if with_optimizer:
            self.optimizer.load_state_dict(save['optimizer'])


    def set_eval_mode(self):
        '''Set agent to evaluation mode'''
        self.is_training = False
        self.model.eval()


    def set_train_mode(self):
        '''Set agent to training mode'''
        self.is_training = True
        self.model.train()


    def preprocess(self, input_state):
        '''
        Preprocess input state.

        Takes a list/ndarray/Tensor of shape : (stacking, height, width, channels)
            1. Transform to Grayscale
            2. Resize 
            3. Normalize
        Output shape : (batch, stacking, height, width)
        '''
        # Map input to tensor
        if not isinstance(input_state, np.ndarray):
            input_state = np.array(input_state)
        if not(torch.is_tensor(input_state)):
            input_state = torch.from_numpy(input_state.copy()).float().to(self.device)
        # Change dim from (stacking, height, width, channels) to (stacking, channels, height, width)
        input_state = input_state.permute(0,3,1,2)
        # Set of transformations on the input state
        transforms = nn.Sequential(
            T.transforms.Grayscale(num_output_channels=1),
            T.transforms.Resize(self.state_space),
            # WARNING : Might not be the good way to do it 
            # Consider using T.ToTensor(), T.ToPILImage() and then compute mean/std
            T.transforms.Normalize((0.5,), (0.5,))
        )
        # Apply transformations
        input_state = transforms(input_state)
        # Remove channel dimension as it is Grayscale and replace it with stacking dimension
        input_state = input_state.squeeze(1) # Remove channel dim (Grayscale so no one cares, if someone does pls send me email)
        # Add one dimension for batch -> (batch, stacking, height, width)
        input_state = input_state.unsqueeze(0)
        return input_state


    def select_action(self, state):
        '''Choose the best action according to DNN prediction'''
        rand = random.random()
        #Exploration with decay (set min_eps = eps for e-greedy exploration)
        if self.is_training:
            self.eps = self.eps * self.dcr
            self.eps = max(self.eps, self.min_eps) #Minimum
        # Best action
        if not(self.is_training) or rand > self.eps:
            # Use model to predict best Q-values 
            # Then choose action associated to the best one
            with torch.no_grad():
                return torch.argmax(self.model(state)).item()
        # Random action
        else:
            return random.randrange(self.action_space)


    def train(self):
        '''One training step using agent's memory'''
        # Make sure the memory is big enough
        if len(self.memory) < self.batch_size:
            return
        
        # Counter to update target network
        self.nb_train += self.batch_size

        # Get random sample of transitions
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch
        # array of Transitions into Transition of arrays
        batch = ReplayMemory.Template()(*zip(*transitions))
        
        # Extract each value in the Transition object
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        # For next_state, we have to remove final states (None values)
        next_states = torch.cat([state for state in batch.next_state if state is not None])
        next_states_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.next_state)), dtype=torch.bool, device=self.device)

        # Compute Q-values and keep the ones associated to the stored actions
        q_pred = self.model(states).gather(1, actions)

        # Initialize Q-values of next states to 0 and compute values for non final ones
        q_next = torch.zeros(self.batch_size, device=self.device)
        q_next[next_states_mask] = self.target_model(next_states).max(1)[0].detach() #Get max from pred
        
        # Compute loss
        q_target = (rewards + self.gamma * q_next).unsqueeze(1)
        loss = self.criterion(q_pred,q_target).to(self.device)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target network update
        # Hard method
        #if self.nb_train > self.target_update:
        #    self.target_model.load_state_dict(self.model.state_dict()) 
        #    self.nb_train = 0
        # Soft method
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.target_ur*model_param.data + (1.0 - self.target_ur)*target_param.data)





# Load environment
env = gym.make(ENV_NAME)
# Transform observation to observation['pov']
# In this case, we need only the POV observation
# and it is mandatory to have Box type for frame stacking
env = CustomObservation(env)
# Stack the STACKING last frames together
env = FrameStack(env,STACKING)


# List of available actions
# Keep only the useful movements for this environment
action_keys = [
    'right',
    'left',
    'attack'
] 


#Size of desired input image (grayscale format)
state_space = (64,64)

#Number of available actions
action_space = len(action_keys)


# Convert an action index (model prediction output) to understandable action for gym
def ind_to_action(ind):
    key = action_keys[ind] # Get the action key from index
    action = env.action_space.noop() # Init an action dict
    action[key] = 1 # Set to 1 as it is binary activation
    action['camera'] = 0,0 # Needed for rendering
    return action


# Dynamically plot training evolution
def plot(interactions,rewards):
    plt.clf() # Clear
    plt.title("Evolution du training")
    plt.plot(interactions, rewards)
    plt.ylabel('Cumul des rewards')
    plt.xlabel('Nb interactions')
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.pause(0.001) # to update info we need to pause a little bit
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# To control the randomness
np.random.seed(SEED)
env.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def train():
    '''Train an agent on Mineline-v0 environment'''
    # DQN Agent 
    agent = DQNAgent(state_space, action_space, STACKING, device=device)

    # To evaluate the training
    rewards = []
    interactions = []
    nb_interactions = 0

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()

    # Early stopping -> when agent is considered as trained 
    # 99 is the maximum obtainable in best configurations, 80 is OK
    best_perfs = 80
    best_perfs_count = 0
    best_perfs_stop = 30
    
    # Run the simulation NB_EPISODES times (upper limit if no early stopping)
    for i_episode in range(NB_EPISODES):
    
        # Reset the environment
        obs = env.reset()

        # Preprocess state image
        state = agent.preprocess(obs)

        # Total of reward during the episode 
        tot_ep_reward = 0
    
        # Run the actions of the agent
        done = False
        while not done:
            # Display
            if RENDER_MODE_TRAIN:
                env.render()
        
            # The agent selects the action and the environment performs it
            action_ind = agent.select_action(state)
            next_obs, reward, done, info = env.step(ind_to_action(action_ind))

            # Preprocess next state image
            next_state = agent.preprocess(next_obs) if not done else None

            # Store the transition in memory
            action = torch.tensor([[action_ind]], device=device)
            reward = torch.tensor([reward], device=device)
            agent.memory.push(state, action, next_state, reward)

            # Update state
            state = next_state

            # Update info to evaluate training
            nb_interactions += 1
            tot_ep_reward += int(reward.cpu()[0])

            # Training
            agent.train()
        
        # Result of the episode
        print(f"Episode {i_episode}, score:{tot_ep_reward}, exploration_r: {round(agent.eps,4)}")

        # Save result of an episode to evaluate training
        rewards.append(tot_ep_reward)
        interactions.append(nb_interactions)

        # Interactive plotting
        plot(interactions,rewards)

        # Early stopping
        # Stop when X good perfs in a row
        if tot_ep_reward >= best_perfs:
            best_perfs_count += 1
        elif best_perfs_count > 0:
            best_perfs_count = 0
        if best_perfs_count >= best_perfs_stop:
            break

        # Save model each 10 episodes
        if i_episode % 10 == 0:
            agent.save_model(path=f'model_save_stacking_{STACKING}.pth')

    # Save last perfs
    agent.save_model(path=f'models/model_stacking_{STACKING}_episodes_{i_episode}_act_{best_perfs_stop}.pth')

    # Stop dynamic plotting and show final graph
    plt.ioff()
    plt.show()
    plt.title("Evolution du training")
    plt.ylabel('Cumul des rewards')
    plt.xlabel('Nb interactions')
    plt.plot(interactions, rewards)
    plt.savefig(f'res/score_stacking_{STACKING}_episodes_{i_episode}_act_{best_perfs_stop}.png')

    # Custom env error, not our fault :(
    try:
        env.close()
    except Exception as e:
        pass 



def test(nb_episodes):
    '''Test an agent on Mineline-v0 environment'''
    # Please see train function for explanation of the code
    video_recorder = VideoRecorder(env=env,path="res/demo.mp4")
    agent = DQNAgent(state_space, action_space, STACKING, device=device)
    agent.load_model(path=f'model_stacking_{STACKING}.pth')
    agent.set_eval_mode()
    tot_reward = 0
    rewards = []
    for i_episode in range(nb_episodes):
        obs = env.reset()
        state = agent.preprocess(obs)
        tot_ep_reward = 0
        done = False
        while not done:
            if RENDER_MODE_TEST:
                env.render()            
            if RECORD_ALL or i_episode+1 == nb_episodes:
                video_recorder.capture_frame()
            action_ind = agent.select_action(state)
            next_obs, reward, done, info = env.step(ind_to_action(action_ind))
            state = agent.preprocess(next_obs) if not done else None
            tot_ep_reward += reward
        print(f"Episode {i_episode}, score:{tot_ep_reward}")
        tot_reward += tot_ep_reward
        rewards.append(tot_ep_reward)
    print(f"Moyenne des recompenses sur {nb_episodes} episodes : {tot_reward/nb_episodes}")
    print(f"Minimum des recompenses sur {nb_episodes} episodes : {min(rewards)}")
    print(f"maximum des recompenses sur {nb_episodes} episodes : {max(rewards)}")
    try:
        env.close()
    except Exception as e:
        pass
    video_recorder.close()
    video_recorder.enabled = False






# ----------------------------------------------------------------------------- #
# ---------------------------     Main       ---------------------------------- #
# ----------------------------------------------------------------------------- #

def usage():
    print(f"Usage : <executable> [ --train OR --test=<nb_episodes> ]")

def main(argv):
    try:   
        opts,_ = getopt.getopt(argv, '', [ 'train', 'test='])
        if (len(opts) == 0):
            raise ValueError()
        for (opt,val) in opts:
            if opt == "--train":
                train()
                return
            elif opt == "--test" and int(val):
                test(int(val))
                return
            else:
                raise ValueError()
    except (getopt.GetoptError, ValueError) as e:
            print("Error : Impossible de lancer le programme...")
            usage()
            sys.exit(2)



if __name__ == '__main__':
    main(sys.argv[1:])
