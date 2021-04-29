# Name: NAME
# Dimensions: 2

import inspect
import math
import random
# --- Set up executable path, do not edit ---
import sys

import numpy as np

this_file_loc = (inspect.stack()[0][1])
main_dir_loc = this_file_loc[:this_file_loc.index('ca_descriptions')]
sys.path.append(main_dir_loc)
sys.path.append(main_dir_loc + 'capyle')
sys.path.append(main_dir_loc + 'capyle/ca')
sys.path.append(main_dir_loc + 'capyle/guicomponents')

# ---

import capyle.utils as utils
from capyle.ca import Grid2D, Neighbourhood, randomise2d

global water_counter
water_counter = 0

DIRECTIONS = {"NW", "N", "NE", "W", "E", "SW", "S", "SE"}
CHAPARRAL, DENSE_FORREST, LAKE, CANYON, BURNING, BURNT, START_BURN, END_BURN = range(
    8)

global initial_grid
base_initial_grid = np.zeros((20, 20), dtype=int)
base_initial_grid[12:16, 6:10] = 1  # adding forrest
base_initial_grid[4:6, 2:8] = 2  # adding lake
base_initial_grid[2:14, 13] = 3 # adding canyon
scale = 10
grid_size = 20 * scale
initial_grid = np.kron(base_initial_grid, np.ones((scale, scale)))
initial_grid[0, (len(initial_grid) - 1)] = 4 #fire starting point

global ignition_grid
ignition_grid = np.zeros((grid_size, grid_size))

generation = 0.5
cell_size = 0.25
chaparral_decay_km = 168
forest_decay_km = 600
canyon_decay_km = 8

chaparral_decay = chaparral_decay_km * cell_size * (1 / generation)
forest_decay = forest_decay_km * cell_size * (1 / generation)
canyon_decay = canyon_decay_km * cell_size * (1 / generation)
global decay_values
decay_values = [
    chaparral_decay, forest_decay, 1, canyon_decay, chaparral_decay
]


def r_value(
        combust_index, wind_speed, psi
):  # wind speed as m/s, psi is angle between wind dir & fire dir,
    # temp in celcius

    temp = 35  #highest temp in texas in july
    humidity = 70  #average texas humidty (most forest fires there). refine later
    w = math.floor(wind_speed / 0.836)**(2 / 3)  #wind level

    kp = math.exp(0.1783 * wind_speed * math.cos(psi))
    r0 = 0.03 * temp + 0.05 * w + 0.01 * (100 - humidity) - 0.3

    r = r0 * kp * combust_index

    return r


def next_state(current_state, neighbours,
               l):  # l = cell size. not sure what this means
    neighbours_states = 0
    r_max = 0
    for neighbours in neighbours:
        this_r_value = 0  # + r_value(neighbour) -- put in info for each neighbour
        if this_r_value > r_max:
            r_max = this_r_value
        neighbour_states = neighbour_states  # + this_r_value
    delta_t = (1 / 8)(l / r_max)

    new_state = (current_state + neighbour_states * delta_t) / l

    return new_state


def setup(args):
    """Set up the config object used to interact with the GUI"""
    # chaparral,denseForest,lake,canyon,burning,burnt = neighbours
    config_path = args[0]
    config = utils.load(config_path)
    # -- THE CA MUST BE RELOADED IN THE GUI IF ANY OF THE BELOW ARE CHANGED --
    config.title = "Forest Fire"
    config.dimensions = 2
    config.states = \
    (
        CHAPARRAL,
        DENSE_FORREST,
        LAKE,
        CANYON,
        BURNING,
        BURNT,
        START_BURN,
        END_BURN
    )

    # ------------  -------------------------------------------------------------

    config.state_colors = \
        [
            (0.6,0.6,0),      #chaparral
            (0,0.4,0),        #dense forrest
            (0,0.5,1),        #lake
            (0.5,0.5,0.5),    #canyon
            (1,0,0),          #burning
            (0.25,0.25,0.25), #burnt
            (1,0.7,0),        #starting to burn
            (0.8,0,0.2)       #ending burn
        ]

    config.grid_dims = (grid_size, grid_size)
    config.num_generations = 1000
    config.set_initial_grid(initial_grid)
    config.wrap = False

    # --------------------------------------------------------------------

    # the GUI calls this to pass the user defined config
    # into the main system with an extra argument
    # do not change
    if len(args) == 2:
        config.save()
        sys.exit()
    return config


def transition_function(grid, neighbourstates, neighbourcounts, decay_grid,
                        water_decay_grid):
    """function that transitions cells in the grid to the next state.
    the function runs through several processes to
    transition the states correctly. calling calling the ignite function
    on each cell. After this this is used to see by how much the ignition
    grid is incremented.
    Args:
        grid: the grid of states representing the forrest to be transitioned
        neighbourstates: the neighbouring states of each of the cells
                         this is a 2d array with an array for each direction
                         from the cell. (N, NE, NW, etc)
        neighbourcounts: an array of arrays for each cell which gives the counts
                         of each of the states neighbouring the cell
        decay_grid:      grid of values which decrease by 1 for each generation
    """

    global water_counter
    global ignition_grid
    neighbourstates = np.array(neighbourstates)
    init_grid = initial_grid.astype(int)
    ig_grid = np.array(ignition_grid)
    windspeed_ignition_modifiers = wind_speed_rvalue("N", 20)
    new_ig_grid = []
    for i, row in enumerate(grid):
        new_ig_grid.append([
            ignite(cell, neighbourstates[:, i, j],
                   windspeed_ignition_modifiers) for j, cell in enumerate(row)
        ])
    new_ig_grid = np.array(new_ig_grid)
    started_to_burn = []
    for i, row in enumerate(grid):
        started_to_burn.append([
            started_burning(cell, ig_grid[i, j], new_ig_grid[i, j])
            for j, cell in enumerate(row)
        ])
    grid[started_to_burn] = START_BURN
    ig_grid = np.add(new_ig_grid, ig_grid)
    full_burn = []
    for i, row in enumerate(grid):
        full_burn.append([
            fully_burning(cell, ig_grid[i, j], decay_grid[i, j])
            for j, cell in enumerate(row)
        ])
    grid[full_burn] = BURNING
    end_burning = []
    for i, row in enumerate(grid):
        end_burning.append([
            ending_burn(cell, decay_grid[i, j], decay_values[int(
                initial_grid[i, j])]) for j, cell in enumerate(row)
        ])
    grid[end_burning] = END_BURN
    decay_grid[(grid == BURNING) | (grid == END_BURN)] -= 1
    burnt_out = (decay_grid == 0)  # find those which have decayed to 0
    grid[(decay_grid == 0
          )] = BURNT  #set all that have decayed to zero to BURNT(7)
    water_counter += 1

    if (water_counter == 100): #time taken for water to dry up
        grid[120:160, 80:120] = initial_grid[120:160, 80:120]
    water_decay_grid[(grid != LAKE)] -= 1  # take one off their decay value
    grid[(water_decay_grid == 0)] = BURNT  # switch their state to 5
    ignition_grid = ig_grid
    return grid


def ignite(cell, neighbours, wind):
    """ generates an ignition factor for a given cell
    the function checks if the cell is eligable to have an ignition factor added
    if it can then the function iterates through each of the cells neighbours
    which have an asociated burning threshhold this number is the multiplied by
    the wind speed modifier. a number is then random generated, if this
    random number is less product of the windspeed modifier and the threshold
    then the ignition factor is increased by the ignition factor multiplied by
    the wind speed modifier. If the cell is in the START_BURN state then
    an adition 10 is added to the ignition factor
    Args:
        cell: the cell in the grid that is having an ignition factor added to it
        neighbours: the neighbour states of the cells
        wind: the windspeed ignition modifiers
    Returns:
        An ignition factor which is generated by the above process. This
        ignition factor determines whether a non burning state starts burning,
        as well as whether a state that has started burning has been completely
        consumed by the fire.
    """

    cell = int(cell)
    ignition_factor = 0
    if cell in [LAKE, BURNING, BURNT, END_BURN]: return ignition_factor
    neighbours = neighbours.astype(int)
    fully_burning_threshhold = [0.04, 0.01, 0, 0.1, 0, 0, 0.04]
    fully_burning_factor = 20
    start_burning_threshhold = [0.02, 0.005, 0, 0.05, 0, 0, 0.04]
    start_burning_factor = 10

    # add to cell ignition factor by multiplying
    # windspeed modifier and the cells burning threshhold
    # if a random number is less than the resulting number add
    # the burning factor multiplied by the wind speed modifier

    for index, neighbour in enumerate(neighbours):
        if neighbour == BURNING:
            if fully_burning_threshhold[cell] * wind[index] >= random.uniform(
                    0, 1):
                ignition_factor += int(
                    math.floor(wind[index] * fully_burning_factor))
        if neighbour in [START_BURN,END_BURN] and \
        start_burning_threshhold[cell] * wind[index] >= random.uniform(0,1):
            ignition_factor += int(
                math.floor(wind[index] * start_burning_factor))

    # if the cell is has already started to burn then a burning factor is
    # automatically applied

    if cell == START_BURN: ignition_factor += start_burning_factor
    return int(ignition_factor)

def started_burning(cell, prev_ig, new_ig):
    """checks whether a state has started to burn"""
    new_ig = int(new_ig)
    prev_ig = int(prev_ig)
    cell = int(cell)
    if cell == START_BURN: return True
    if cell not in [LAKE, BURNING, BURNT, END_BURN]:
        if prev_ig == 0 and new_ig > 0:
            return True
    return False


def fully_burning(cell, new_ig, decay):
    """checks whether a state can transition to a fully burning state
    a state becomes fully burning when its ignition value is greater than
    or equal to its initial decay value.
    """

    new_ig = int(new_ig)
    decay = int(decay)
    cell = int(cell)
    if cell == BURNING: return True
    if cell == START_BURN and new_ig >= decay: return True
    return False


def ending_burn(cell, decay, initial):
    """checks whether a state
    has decayed enough to transition to an end burn state
    a cell is ending its burning phase if its original decay value has halved
    """
    cell = int(cell)
    decay = int(decay)
    initial = int(initial)
    if cell == END_BURN: return True
    if cell == BURNING:
        if initial >= 2 * decay:
            return True
    return False



def wind_speed_rvalue(direction, speed):
    if direction in DIRECTIONS:
        list_directions = np.array(
            ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
        item_index = np.where(list_directions == direction)[0]
        listWeights = np.zeros(8)
        angle_interval = 45
        angle = 0  #initialises weight
        wrapped = False
        for x in range(
                8
        ):  #goes through array, including wrapping round and weights the directions
            listWeights[(x + item_index) % len(list_directions)] = k_wind(
                speed, angle)
            angle = angle + angle_interval
        rearranged_index = [
            7, 0, 1, 6, 2, 5, 4, 3
        ]  #rearranges list so is in same order as the CA programme
        return listWeights[rearranged_index]


def k_wind(speed, angle):
    return np.exp(0.1783 * speed * np.cos(np.deg2rad(angle)))


def main():

    config = setup(sys.argv[1:])
    s = -10000
    decay_grid = [[decay_values[i] for i in row]
                  for row in initial_grid.astype(int)]
    decay_grid = np.array(decay_grid)
    water_decay_values = [s, s, s, s, s]
    water_decay_grid = np.array([[water_decay_values[i] for i in row]
                                 for row in initial_grid.astype(int)])

    #Select section of grid to drop water
    water_decay_grid[120:160, 80:120] = 0  #drop water after this time
    ignition_grid = np.zeros((grid_size, grid_size))
    ignition_grid = ignition_grid.astype(int)
    grid = Grid2D(config, (transition_function, decay_grid, water_decay_grid))

    # Create grid object using parameters from config + transition function
    # Run the CA, save grid state every generation to timeline
    timeline = grid.run()

    config.save()  # Save updated config to file
    utils.save(timeline, config.timeline_path)  # Save timeline to file


if __name__ == "__main__":
    main()
