import pygame, sys, random 
from collections import defaultdict
import numpy as np
import argparse
import csv
import dill
#import pickle

def draw_floor():
    screen.blit(floor_surface,(floor_x_pos,900))
    screen.blit(floor_surface,(floor_x_pos + 576,900))

#TODO: add more variable logic to pipes for more interesting challenges for agent to overcome
def create_pipe():
    random_pipe_pos = random.choice(pipe_height)
    bottom_pipe = pipe_surface.get_rect(midtop = (700,random_pipe_pos))
    top_pipe = pipe_surface.get_rect(midbottom = (700,random_pipe_pos - 300))
    return bottom_pipe,top_pipe

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 5
    visible_pipes = [pipe for pipe in pipes if pipe.right > -50]
    return visible_pipes

def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface,pipe)
        else:
            flip_pipe = pygame.transform.flip(pipe_surface,False,True)
            screen.blit(flip_pipe,pipe)

def check_collision(pipes):
    global can_score
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            death_sound.play()
            can_score = True
            return False

    if bird_rect.top <= -100 or bird_rect.bottom >= 900:
        can_score = True
        return False

    return True

def rotate_bird(bird):
    new_bird = pygame.transform.rotozoom(bird,-bird_movement * 3,1)
    return new_bird

def bird_animation():
    new_bird = bird_frames[bird_index]
    new_bird_rect = new_bird.get_rect(center = (100,bird_rect.centery))
    return new_bird,new_bird_rect

def score_display(game_state):
    if game_state == 'main_game':
        score_surface = game_font.render(str(int(score)),True,(255,255,255))
        score_rect = score_surface.get_rect(center = (288,100))
        screen.blit(score_surface,score_rect)
    if game_state == 'game_over':
        score_surface = game_font.render(f'Score: {int(score)}' ,True,(255,255,255))
        score_rect = score_surface.get_rect(center = (288,100))
        screen.blit(score_surface,score_rect)

        high_score_surface = game_font.render(f'High score: {int(high_score)}',True,(255,255,255))
        high_score_rect = high_score_surface.get_rect(center = (288,850))
        screen.blit(high_score_surface,high_score_rect)

def update_score(score, high_score):
    if score > high_score:
        high_score = score
    return high_score

def pipe_score_check():
    global score, can_score 
    increased = 0
    if pipe_list:
        for pipe in pipe_list:
            if 95 < pipe.centerx < 105 and can_score:
                score += 1
                score_sound.play()
                can_score = False
                increased = 1
            
            if pipe.centerx < 0:
                can_score = True
                increased = 0

    return increased

    # Query agent for action based on state
    # Returns true if agent chooses to flap / jump
    # Returns false if agent chooses to wait
def agent_action(state,epsilon):
    global Q
    # Defining state:
    # Work towards image analysis for Deep Q-Learning
    # To start, use basic params:
    # Bird height relative to next bottom pipe, Bird velocity, x distance to next pipe (back edge)
    # Issue: How to make sure reward doesn't bloat as score increases, 
    #   inflating the value of moves made later in the run?
    # This could unlearn general good behavior in favor of a situationally "very good" (overinflated)
    # Also, still need to punish a poor move mostly independent of 

    
    # Right now, epsilon greedy choice

    if random.uniform(0,1) < (1-epsilon):
        action = np.argmax(Q[state]) 
    else:
        action = random.choice(range(2))
    return action

qout = False
csvout = False

parser = argparse.ArgumentParser(description='Train agent for flappy bird with Q-Table')
parser.add_argument('--q_table_path',required=True,
                    help='PKL output file for trained Q-table')
parser.add_argument('--out_file', required=True,
                    help='CSV output file for training results')
parser.add_argument('--in_file',
                    help='pkl file for starting Q-table')
args = parser.parse_args()

if args.q_table_path is not None:
    outQfile = open(args.q_table_path, "wb")
    qout = True

if args.out_file is not None:
    outfile = open(args.out_file,"w")
    csvwriter = csv.writer(outfile)
    csvout = True

if args.in_file is not None:
    infile = open(args.in_file, 'rb')
    Q = dill.load(infile)
else:
    # Need global Q table that persists across runs, state maps to two actions (jump or wait)
    Q = defaultdict(lambda: [-1,-1])



#pygame.mixer.pre_init(frequency = 44100, size = 16, channels = 2, buffer = 1024)
pygame.init()
screen = pygame.display.set_mode((576,1024))
clock = pygame.time.Clock()
game_font = pygame.font.Font('04B_19.ttf',40)

# Game Variables
gravity = 0.25
bird_movement = 0
game_active = True
score = 0
high_score = 0
can_score = True
bg_surface = pygame.image.load('assets/background-day.png').convert()
bg_surface = pygame.transform.scale2x(bg_surface)
epsilon = .4
alpha = 0.5
n_episodes = 10000
epsilon_inc = 1 / n_episodes
alpha_inc =  (alpha-0.1) / (2*n_episodes)
gamma = 0.95
last_action = 0
reward = 0
agent = False
scale = 20
punished = True
results = []
runCount = 1



floor_surface = pygame.image.load('assets/base.png').convert()
floor_surface = pygame.transform.scale2x(floor_surface)
floor_x_pos = 0

bird_downflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-downflap.png').convert_alpha())
bird_midflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-midflap.png').convert_alpha())
bird_upflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-upflap.png').convert_alpha())
bird_frames = [bird_downflap,bird_midflap,bird_upflap]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center = (100,512))

BIRDFLAP = pygame.USEREVENT + 1
pygame.time.set_timer(BIRDFLAP,200)

# bird_surface = pygame.image.load('assets/bluebird-midflap.png').convert_alpha()
# bird_surface = pygame.transform.scale2x(bird_surface)
# bird_rect = bird_surface.get_rect(center = (100,512))

pipe_surface = pygame.image.load('assets/pipe-green.png')
pipe_surface = pygame.transform.scale2x(pipe_surface)
pipe_list = []
SPAWNPIPE = pygame.USEREVENT
pygame.time.set_timer(SPAWNPIPE,900)
pipe_height = [400,600,800]

game_over_surface = pygame.transform.scale2x(pygame.image.load('assets/message.png').convert_alpha())
game_over_rect = game_over_surface.get_rect(center = (288,512))

flap_sound = pygame.mixer.Sound('sound/sfx_wing.wav')
death_sound = pygame.mixer.Sound('sound/sfx_hit.wav')
score_sound = pygame.mixer.Sound('sound/sfx_point.wav')
score_sound_countdown = 100
SCOREEVENT = pygame.USEREVENT + 2
pygame.time.set_timer(SCOREEVENT,100)

# AGENTEVENT- Query agent for action
AGENTEVENT = pygame.USEREVENT + 3
#pygame.time.set_timer(AGENTEVENT,120)

while True:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            if qout:
                dill.dump(Q, outQfile)
            
            if csvout:
                csvwriter.writerows(results)
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and game_active:
                bird_movement = 0
                bird_movement -= 8
                flap_sound.play()
                last_action = 1
                pygame.time.set_timer(AGENTEVENT,0)
                agent = False
                
            if event.key == pygame.K_SPACE and game_active == False:
                game_active = True
                pipe_list.clear()
                pygame.time.set_timer(SPAWNPIPE,0)
                bird_rect.center = (100,512)
                bird_movement = 0
                score = 0
                reward = 0
                punished = False
                pipe_list.extend(create_pipe())
                pygame.time.set_timer(SPAWNPIPE,900)
            
            if event.key == pygame.K_TAB and agent == True:
                pygame.time.set_timer(AGENTEVENT,0)
                agent = False
            
            if event.key == pygame.K_TAB and agent == False:
                pygame.time.set_timer(AGENTEVENT,120)
                agent = True

        if event.type == SPAWNPIPE:
            pipe_list.extend(create_pipe())

        if event.type == BIRDFLAP:
            if bird_index < 2:
                bird_index += 1
            else:
                bird_index = 0

            bird_surface,bird_rect = bird_animation()
        
        # Query Agent for action, if so have the same effect as a user-activated jump
        if event.type == AGENTEVENT and game_active:
            pipe_distance = 576-bird_rect.centerx
            height_diff = bird_rect.centery
            if (pipe_list):
                pipe_distance = pipe_list[0].right - bird_rect.centerx
                height_diff = pipe_list[0].top - bird_rect.centery
                if(pipe_distance<0):
                    pipe_distance = pipe_list[2].right - bird_rect.centerx
                    height_diff = pipe_list[2].top - bird_rect.centery

            state = (height_diff//scale, bird_movement//1, pipe_distance//scale)
            # Prepare state tuple from heght relative to next pipe, vertical velocity, 
            #   and x distance to next pipe. To limit state space growth, 
            #   the distances are broken up into 10 pixel increments
            last_action = agent_action(state, epsilon)
            if(last_action):
                # If chosen move is to flap, then flap
                bird_movement = -7
                flap_sound.play()
            
        
        # Game not active yet, need to activate it to get the agent to start interacting with the game
        if event.type == AGENTEVENT and not game_active:
            game_active = True
            pipe_list.clear()
            pygame.time.set_timer(SPAWNPIPE,0)
            bird_rect.center = (100,512)
            bird_movement = 0
            reward = 0
            score = 0
            punished = False
            pipe_list.extend(create_pipe())
            pygame.time.set_timer(SPAWNPIPE,900)
            
    screen.blit(bg_surface,(0,0))
    # State should be unchanged, but confirm in case weird edge case
    pipe_distance = 576-bird_rect.centerx
    height_diff = bird_rect.centery
    if (pipe_list):
        
        pipe_distance = pipe_list[0].right - bird_rect.centerx
        height_diff = pipe_list[0].top - bird_rect.centery
        if(pipe_distance<0):
            pipe_distance = pipe_list[2].right - bird_rect.centerx
            height_diff = pipe_list[2].top - bird_rect.centery
    distance_line = pygame.draw.line(screen, pygame.Color(255,0,0), (bird_rect.centerx, bird_rect.centery), (bird_rect.centerx+pipe_distance,bird_rect.centery))
    height_line = pygame.draw.line(screen, pygame.Color(0,0,255), (bird_rect.centerx, bird_rect.centery), (bird_rect.centerx,bird_rect.centery+height_diff))
        

    state = (height_diff//scale, bird_movement//1, pipe_distance//scale)
    

    # This is where things actually happen in the game, bird moves up/down,
    #   pipes move 5 pixels closer to the bird, score is updated
    if game_active:
        # Bird
        bird_movement += gravity
        rotated_bird = rotate_bird(bird_surface)
        bird_rect.centery += bird_movement
        screen.blit(rotated_bird,bird_rect)
        game_active = check_collision(pipe_list[:2])

        # Pipes
        pipe_list = move_pipes(pipe_list)
        draw_pipes(pipe_list)
        
        # Score
        # If we get a score increase, want the immediate reward to reflect that
        pipe_score_check()
        reward = 0
        score_display('main_game')
        next_state = (height_diff//scale, bird_movement//1, pipe_distance//scale)

        # Check new state context
        pipe_distance = 576-bird_rect.centerx
        height_diff = bird_rect.centery
            
        if (pipe_list):
            pipe_distance = pipe_list[0].right - bird_rect.centerx
            height_diff = pipe_list[0].top - bird_rect.centery
            if(pipe_distance<0):
                pipe_distance = pipe_list[2].right - bird_rect.centerx
                height_diff = pipe_list[2].top - bird_rect.centery
                #reward -= 1

        next_state = (height_diff//scale, bird_movement//1, pipe_distance//scale)
        
        # Update Q table
        prev_Q = Q[state][last_action]
        Q[state][last_action] =  prev_Q + alpha*(reward + gamma*np.max(Q[next_state])- prev_Q)
        


    # Game has ended, we died, punish agent
    if not game_active:
        screen.blit(game_over_surface,game_over_rect)
        high_score = update_score(score,high_score)
        score_display('game_over')

        

        reward = -100
        if not punished:
            punished = True

            # Run ended
            results.append([runCount,score, agent])
            runCount += 1

            # Check new state context
            pipe_distance = 576-bird_rect.centerx
            height_diff = bird_rect.centery
            
            if (pipe_list):
                pipe_distance = pipe_list[0].centerx - bird_rect.centerx
                height_diff = pipe_list[0].top - bird_rect.centery

            next_state = (height_diff//scale, bird_movement//1, pipe_distance//scale)

             # Update Q table
            prev_Q = Q[state][last_action]
            Q[state][last_action] =  prev_Q + alpha*(reward + gamma*np.max(Q[next_state])- prev_Q)

            alpha -= alpha_inc
            alpha = min(0.1, alpha)
            # Decrement random exploration
            epsilon -= epsilon_inc
            

    # Do nothing by default
    last_action = 0

    # Floor
    floor_x_pos -= 1
    draw_floor()
    if floor_x_pos <= -576:
        floor_x_pos = 0
    
    #print("height diff: " + str(height_diff))
    #print("pipe_distance: " + str(pipe_distance))
    

    pygame.display.update()
    clock.tick(120)
