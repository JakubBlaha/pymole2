##!/usr/bin/env python
import pygame
from pygame.locals import *
import numpy as np

fsize = 32
WALL_NUMBER      = 57
BOMB_ITEM_NUMBER = 6
FIRE_ITEM_NUMBER = 6
TOOTHER_NUMBER   = 2
GHOST_NUMBER     = 2
MONSTER_MIND     = 6  # the higher number the less often monster turns
BOMB_TIMER       = 40 # it takes 40 frames bomb to explode
PLAN_SIZE        = (15,11)

def load_images(fname):
 images = pygame.image.load(fname).convert_alpha()
 w, h = images.get_size()
 return np.array([[images.subsurface((i*fsize, j*fsize, fsize, fsize)) for j in range(h//fsize)] for i in range(w//fsize)])

###
# Player class defines players behaviour
###

class Player:
  animation_frames=8
  dying_animation_frames=12 
  grid=5
  field_span = np.array([0, grid-1])
  
  def __init__(self, pos, keys, sprites):
    self.phase = 0
    self.orientation = 0
    self.posx, self.posy = pos[0]*self.grid, pos[1]*self.grid
    self.kup, self.kdown, self.kleft, self.kright, self.kbomb = keys
    self.sprites = sprites
    self.max_bombs = 1
    self.bomb_range = 2
    self.my_bombs = []

  def step(self, pressed_keys):
    in_pressed_keys = lambda kset: len(kset & pressed_keys) > 0
    pos_in_plan = (self.posx+self.grid//2)//self.grid, (self.posy+self.grid//2)//self.grid

    if plan[pos_in_plan] == item_id: # Pick up an item
      if plan_data[pos_in_plan] == 1:
        self.max_bombs +=1
      else:
        self.bomb_range +=1
      plan[pos_in_plan] = floor_id
    
    killing_monster = len([m for m in monsters if abs(self.posx*m.grid-m.posx*self.grid)+abs(self.posy*m.grid-m.posy*self.grid) < m.grid*self.grid//2])

    if (plan[pos_in_plan] == fire_id or killing_monster) and (self.orientation != 4 or self.phase == 0): # Player is hit by fire or monster?
      self.orientation = 4 
      self.phase = 0
    elif self.orientation == 4: # player is already dying
      self.phase += 1
      if self.phase == self.dying_animation_frames: # player is completely dead 
        players.remove(self)
        return
    else:
      self.my_bombs = [bomb_pos for bomb_pos in self.my_bombs if plan[bomb_pos] == bomb_id]
      if len(self.my_bombs) < self.max_bombs and in_pressed_keys(self.kbomb) and not plan[pos_in_plan]: # Drop a bomb?
        self.my_bombs.append(pos_in_plan)
        plan[pos_in_plan] = bomb_id
        plan_phase[pos_in_plan] = BOMB_TIMER
        plan_data[pos_in_plan] = self.bomb_range
        pressed_keys -= self.kbomb
      
      # The rest of the methods takes care of player movement
      stepx = in_pressed_keys(self.kright) - in_pressed_keys(self.kleft)
      stepy = in_pressed_keys(self.kdown)  - in_pressed_keys(self.kup)
      if stepx or stepy: 
        self.phase=(self.phase+1) % self.animation_frames
        self.orientation = 2 if stepx > 0 else 1 if  stepx < 0 else (stepy<0)*3
      else:
        self.phase=0

      blocked_corners=plan[np.ix_((self.posx+stepx+self.field_span)//self.grid, (self.posy+stepy+self.field_span)//self.grid)]>item_id
      corner_xing_field = np.logical_or.outer(np.logical_and(not self.posx%self.grid, [stepx<0, stepx>0]),
                                              np.logical_and(not self.posy%self.grid, [stepy<0, stepy>0]))
      blocked_corners = np.logical_and(blocked_corners, corner_xing_field)                                        
      if stepx and stepy and blocked_corners.sum() == 1: #steping diagonally against corner
        if self.posx % self.grid or not (self.posy % self.grid or plan[self.posx//self.grid+stepx,self.posy//self.grid]):
          stepy = 0
        else:
          stepx = 0
      else:
        stepx += np.sign(np.sum(blocked_corners.T*[1,-1])) # step back in the blocked directiones
        stepy += np.sign(np.sum(blocked_corners  *[1,-1]))
      if (stepx or stepy) and not (blocked_corners.sum() == 2 and np.all(blocked_corners==blocked_corners.T)): 
        # We are making step AND NOT through blocks touching by corners (i.e. blocked_corners is not diagonal)
        self.posx += stepx
        self.posy += stepy
        self.orientation = 2 if stepx > 0 else 1 if  stepx < 0 else (stepy<0)*3

    screen.blit(self.sprites[self.orientation*4+self.phase//2], (self.posx*fsize//self.grid, self.posy*fsize//self.grid-fsize//2))

class Monster:
  animation_frames=8
  dying_animation_frames=12 
  grid=10
  field_span = np.array([0, grid-1])
  
  def __init__(self, pos, ghost=0):
    self.phase = 0
    self.posx, self.posy = pos[0]*self.grid, pos[1]*self.grid
    self.stepx, self.stepy = 0, 0
    self.ghost = ghost
    self.sprites = i_monsters[ghost]

  def step(self, pressed_keys):
    x, y = (self.posx+self.grid//2)//self.grid, (self.posy+self.grid//2)//self.grid # position in the plan

    if plan[x, y] == fire_id: # monster is hit by fire
      self.phase = self.animation_frames
    elif self.phase >= self.animation_frames: # monster is already dying
      self.phase += 1
      if self.phase == self.animation_frames+self.dying_animation_frames: # monster is completely dead 
        monsters.remove(self)
        return
    else:
      target_field_blocked = plan[(self.posx+self.stepx+self.grid*(self.stepx==1))//self.grid, 
                                  (self.posy+self.stepy+self.grid*(self.stepy==1))//self.grid] >= wall_id+self.ghost
      field_centered = not (self.posx % self.grid or self.posy % self.grid)
      if not field_centered and target_field_blocked: # Somebody just blocked the way with a bomb. Turn back!
        self.stepx = -self.stepx
        self.stepy = -self.stepy
      elif (self.stepx + self.stepy == 0 or field_centered
          and (target_field_blocked or not np.random.randint(MONSTER_MIND))): # or we just randomly decided to turn
        possible_steps = np.nonzero(plan[([x-1,x+1, x, x], [y, y, y-1, y+1])] < wall_id+self.ghost)[0]
        newdir = np.random.choice(possible_steps) if len(possible_steps) else -1 # choose new direction
        self.stepx = int(newdir == 1) - int(newdir == 0)
        self.stepy = int(newdir == 3) - int(newdir == 2)
      self.phase=(self.phase+1) % self.animation_frames
      self.posx += self.stepx
      self.posy += self.stepy
    screen.blit(self.sprites[self.phase//2], (self.posx*fsize//self.grid, self.posy*fsize//self.grid-fsize//2))

pygame.init()
screen = pygame.display.set_mode((PLAN_SIZE[0]*32, (PLAN_SIZE[1]-1)*32), pygame.FULLSCREEN)
clock = pygame.time.Clock()
pygame.joystick.init()
joystick_count = pygame.joystick.get_count()
joysticks = [pygame.joystick.Joystick(i) for i in range(joystick_count)]
for joystick in joysticks:
  joystick.init()

#pygamekeys = {key: value for key, value in pygame.locals.__dict__.items() if key.startswith('K_')}

i_abomb, i_afire, i_bomb, i_floor, i_rock = load_images("items.png")[0]
i_walls = load_images("walls.png")[0]
i_players = load_images("players.png")
i_monsters = load_images("monsters.png")
i_fires = load_images("fires.png")
i_explosion = load_images("explosion.png")[0]
i_bombs = load_images("bombs.png")[0]
pygame.display.set_icon(i_bomb)
pygame.display.set_caption("PyMole")

fire_id  = -1
floor_id =  0
item_id  =  1 # 1 and higher block fire
wall_id  =  2 # 2 and higher players and fire
rock_id  =  3 # 3 and higher block ghosts
bomb_id  =  4
wall_frames = len(i_walls)*2
explosion_frames = len(i_explosion)*2
fire_frames = i_fires.shape[1]*3 # how namy frames takes explosion

game_over = False
while not game_over:
   # Init plan
  plan = np.zeros(PLAN_SIZE, dtype=int)
  plan[[0, -1],:] = rock_id
  plan[:,[0, -1]] = rock_id
  plan[0::2,0::2] = rock_id
  plan_phase = np.zeros_like(plan, dtype=int) # phase of fire, explosion, collapsing wall, bomb to explode
  plan_data  = np.zeros_like(plan, dtype=int) # strength of bomb, orientation of fire

  wall_possible = np.zeros_like(plan, dtype=bool)
  wall_possible[3:-3,:] = True
  wall_possible[:,3:-3] = True
  wall_possible[plan!=0] = False
  wall_positiones=np.random.choice(np.where(wall_possible.flat)[0], min(wall_possible.sum(), WALL_NUMBER+TOOTHER_NUMBER+GHOST_NUMBER), replace=False)
  monster_positiones = np.unravel_index(wall_positiones[:TOOTHER_NUMBER+GHOST_NUMBER], plan.shape)
  wall_positiones = wall_positiones[TOOTHER_NUMBER+GHOST_NUMBER:]
  plan.flat[wall_positiones] = wall_id
  plan_data.flat[wall_positiones[:BOMB_ITEM_NUMBER]]  = 1 # hide items under some of the walls
  plan_data.flat[wall_positiones[-FIRE_ITEM_NUMBER:]] = 2

  # Create players and monsters
  monsters = [Monster(pos, ghost=int(i>=TOOTHER_NUMBER)) for i,pos in enumerate(zip(*monster_positiones))]
  players =  [Player(plan.shape-np.array(2), ({K_UP, (0, 0, -1)}, {K_DOWN, (0, 0, 1)},  {K_LEFT, (0, 1, -1)}, {K_RIGHT, (0, 1, 1)}, {K_RCTRL, (0, 11)}), i_players[0]),
              Player((1,1),                  ({K_w,  (0, 3)},     {K_s, (0, 1)},        {K_a, (0, 4)},        {K_d, (0, 0)},        {K_c, (0, 10)}),     i_players[1]),
              Player((1, plan.shape[1]-2),   ({K_t, (1, 0, -1)},  {K_g, (1, 0, 1)},     {K_f, (1, 1, -1)},    {K_h, (1, 1, 1)},     {K_n, (1, 8)}),      i_players[2]),
              Player((plan.shape[0]-2, 1),   ({K_i, (1, 0)},      {K_k, (1, 2)},        {K_j, (1, 3)},        {K_l, (1, 1)},        {K_PERIOD, (1, 9)}), i_players[3]),]

  pressed_keys = set()
  # Keep playing untill there is only one player left
  while not game_over and len(players) > 1:
      for event in pygame.event.get():
#          print(event)
          if event.type == pygame.locals.QUIT:
              game_over = True
          elif event.type == pygame.KEYDOWN:
            pressed_keys.add(event.key)
          elif event.type == pygame.KEYUP:
            pressed_keys.discard(event.key)
          elif event.type == pygame.JOYBUTTONDOWN:
            pressed_keys.add((event.joy, event.button))
          elif event.type == pygame.JOYBUTTONUP:
            pressed_keys.discard((event.joy, event.button))
          elif event.type == pygame.JOYAXISMOTION:
            position = round(event.value)
            pressed_keys -= {(event.joy, event.axis, -1), (event.joy, event.axis, 1)}
            if position:
              pressed_keys.add((event.joy, event.axis, position))

      # explode bombs that run out of time
      for x, y in zip(*np.nonzero(np.logical_and(plan == bomb_id, plan_phase == 0))):
                hit_fields = []
                bomb_range = plan_data[x,y]
                blocking_fire = np.nonzero(plan[:,y] > 0)[0]
                h_fire_range = range(max(x-bomb_range,   blocking_fire[blocking_fire < x][-1]+1),
                                     min(x+bomb_range+1, blocking_fire[blocking_fire > x][0]))
                plan[h_fire_range,y]=fire_id
                plan_phase[h_fire_range,y]=0
                plan_data[h_fire_range,y]=3
                if h_fire_range.start == x-bomb_range: 
                  plan_data[h_fire_range.start,y] = 5
                else:
                  hit_fields.append((h_fire_range.start-1,y))
                if h_fire_range.stop == x+bomb_range+1: 
                  plan_data[h_fire_range.stop-1,y] = 4
                else:
                  hit_fields.append((h_fire_range.stop,y))

                blocking_fire = np.nonzero(plan[x,:] > 0)[0]
                v_fire_range = range(max(y-bomb_range,   blocking_fire[blocking_fire < y][-1]+1),
                                     min(y+bomb_range+1, blocking_fire[blocking_fire > y][0]))
                plan[x, v_fire_range]=fire_id
                plan_phase[x, v_fire_range]=0
                plan_data[x, v_fire_range]=0
                if v_fire_range.start == y-bomb_range: 
                  plan_data[x,v_fire_range.start] = 1
                else:
                  hit_fields.append((x,v_fire_range.start-1))
                if v_fire_range.stop == y+bomb_range+1: 
                  plan_data[x,v_fire_range.stop-1] = 2
                else:
                  hit_fields.append((x,v_fire_range.stop))

                plan[x,y] = fire_id
                plan_phase[x, y]=0
                plan_data[x, y]=6

                if hit_fields: # explode bombs, items and walls hit by this bomb
                  plan_phase[tuple(zip(*hit_fields))] = 1
                    
      # Draw and change state of all the fields in the plan
      for x, row in enumerate(plan):
        for y,  field in enumerate(row):
            tile = i_floor
            if   field == rock_id:
              tile = i_rock
            elif field == bomb_id:
              tile = i_bombs[(BOMB_TIMER-plan_phase[x,y])*len(i_bombs)//BOMB_TIMER]
              plan_phase[x,y] -= 1
            elif field == fire_id:
              plan_phase[x,y] += 1
              if plan_phase[x,y] < fire_frames: # explosion is not over
                tile = i_fires[plan_data[x,y], plan_phase[x,y]*i_fires.shape[1]//fire_frames]
              else:
                plan[x,y] = floor_id
                #tile = i_floor
            elif field == wall_id:
              if plan_phase[x,y] == 0: # wall is intact
                tile = i_walls[0]
              elif plan_phase[x,y] == wall_frames: # wall has just completely collapsed
                plan[x,y] = item_id if plan_data[x,y] != 0 else floor_id # Was the wall hidding an item?
                plan_phase[x,y] = 0
                #tile = i_abomb if plan_data[x,y] == 1 else i_afire
              else: # wall is colapsing
                tile = i_floor if plan_data[x,y] == 0 else i_abomb if plan_data[x,y] == 1 else i_afire
                screen.blit(tile, (x*fsize, y*fsize-16))
                tile = i_walls[plan_phase[x,y]*len(i_walls)//wall_frames]
                plan_phase[x,y] += 1
            
            if plan[x,y] == item_id:
              if plan_phase[x,y] == explosion_frames: # item has just completely burned
                plan[x,y] = floor_id
                #tile = i_floor
              else: 
                tile = i_abomb if plan_data[x,y] == 1 else i_afire
                if plan_phase[x,y] > 0: # item is burning
                  screen.blit(tile if plan_phase[x,y] < explosion_frames*2//3 else i_floor, (x*fsize, y*fsize-16))
                  tile = i_explosion[plan_phase[x,y]*len(i_explosion)//explosion_frames]
                  plan_phase[x,y] += 1            
            screen.blit(tile, (x*fsize, y*fsize-16))
      if len(players) < 2:
        break
      for p in monsters+players:
        p.step(pressed_keys)
        
      pygame.display.flip()
      clock.tick(15)
