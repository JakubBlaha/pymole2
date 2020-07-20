import pygame
import pygame.locals
import numpy as np

floor_id = 0
rock_id  = 1
wall_id  = 2
fsize = 32

WALL_NUMBER      = 20 #57
BOMB_ITEM_NUMBER = 6
FIRE_ITEM_NUMBER = 6
TOOTHER_NUMBER   = 2
GHOST_NUMBER     = 2

def load_images(fname):
 images = pygame.image.load(fname).convert_alpha()
 w, h = images.get_size()
 return np.array([[images.subsurface((i*fsize, j*fsize, fsize, fsize)) for j in range(h/fsize)] for i in range(w/fsize)])

pygame.init()
screen = pygame.display.set_mode((480, 320))
clock = pygame.time.Clock()


i_abomb, i_afire, i_bomb, i_floor, i_rock = load_images("items.png")[0]
i_walls = load_images("walls.png")[0]
i_players = load_images("players.png")

###
# Init plan
###

plan = np.zeros((15,11))
plan[[0, -1],:] = rock_id
plan[:,[0, -1]] = rock_id
plan[0::2,0::2] = rock_id

wall_possible = np.zeros_like(plan, dtype=bool)
wall_possible[3:-3,:] = True
wall_possible[:,3:-3] = True
wall_possible[plan!=0] = False
plan.flat[np.random.choice(np.where(wall_possible.flat)[0], WALL_NUMBER, replace=False)] = wall_id


player_position = np.array([[1.0, 1.0], 
                            np.array(plan.shape) - 2])
player_phase = 0
player_orientation = 0
step_size=0.25




game_over = False
while not game_over:
    for event in pygame.event.get():
        if event.type == pygame.locals.QUIT:
            game_over = True
   
    pressed_keys = pygame.key.get_pressed()
    
    xcentered = not int(player_position[0][0]*4)%4
    ycentered = not int(player_position[0][1]*4)%4
    player_phase=(player_phase+1)%8

    if xcentered and (
             pressed_keys[pygame.locals.K_DOWN]                    and not (ycentered and plan[tuple((player_position[0]+[ 0,   1]).astype(int))])
          or pressed_keys[pygame.locals.K_RIGHT] and not ycentered and not plan[tuple((player_position[0]+[ 1.5, 1]).astype(int))] 
          or pressed_keys[pygame.locals.K_LEFT]  and not ycentered and not plan[tuple((player_position[0]+[-0.5, 1]).astype(int))] 
         ):
      player_position[0][1] += step_size
      player_orientation=0
    elif  xcentered and (
             pressed_keys[pygame.locals.K_UP]                      and not (ycentered and plan[tuple((player_position[0]-[0,    1  ]).astype(int))])
          or pressed_keys[pygame.locals.K_RIGHT] and not ycentered and not plan[tuple((player_position[0]+[1.5,  0]).astype(int))] 
          or pressed_keys[pygame.locals.K_LEFT]  and not ycentered and not plan[tuple((player_position[0]+[-0.5, 0]).astype(int))] 
         ):
      player_position[0][1] -= step_size
      player_orientation=3
    elif ycentered and (
             pressed_keys[pygame.locals.K_LEFT]                    and not (ycentered and plan[tuple((player_position[0]-[1,    0]).astype(int))])
          or pressed_keys[pygame.locals.K_DOWN]  and not xcentered and not plan[tuple((player_position[0]+[0,  1.5]).astype(int))] 
          or pressed_keys[pygame.locals.K_UP]    and not xcentered and not plan[tuple((player_position[0]+[0, -0.5]).astype(int))] 
         ):
      player_position[0][0] -= step_size
      player_orientation=1
    elif ycentered and (
             pressed_keys[pygame.locals.K_RIGHT]                   and not (ycentered and plan[tuple((player_position[0]+[1,   0]).astype(int))])
          or pressed_keys[pygame.locals.K_DOWN]  and not xcentered and not plan[tuple((player_position[0]+[1, 1.5]).astype(int))] 
          or pressed_keys[pygame.locals.K_UP]    and not xcentered and not plan[tuple((player_position[0]+[1,-0.5]).astype(int))] 
         ):
      player_position[0][0] += step_size
      player_orientation=2
    else:
      player_phase=0

    for x, row in enumerate(plan):
       for y,  field in enumerate(row):
          if   field == rock_id:
            tile = i_rock
          elif field == wall_id:
            tile = i_walls[0]
          else:
            tile = i_floor
          screen.blit(tile, (x*fsize, y*fsize-16))
      
    for p, (x, y) in enumerate(player_position):
      screen.blit(i_players[p][player_orientation*4+player_phase/2], player_position[p]*fsize-[0,16])

    pygame.display.flip()
    clock.tick(15)
