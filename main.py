import retro
import numpy as np
import neat
import cv2
import os
import matplotlib.pyplot as plt
import graphviz
import neat.visualize as visualize

# make environment for game
env = retro.make(game='SonicTheHedgehog2-Genesis', state='ChemicalPlantZone.Act1', scenario='scenario.json', record='.')


# for inputs we would have to get the game screen pixels and multiply them by the number of inputs possible

imgarray = []

x_end = 0
def eval_genomes(genomes, config):
    res_x, res_y, res_dummy = env.observation_space.shape # we get a tuple of 3 elements, only first 2 matter as they are the genesis resolution

    # split res into smaller pieces for simplicity
    res_x //= 8
    res_y //= 8
    
    for genome_id, genome in genomes:
        ob = env.reset() # observation space
        ac = env.action_space.sample() # action space, random sample of random actions

        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        
        current_max_fitness = 0
        current_fitness = 0
        frames = 0
        idle_frames = 0 # after x amount of frames w no improvement, we continue
        
        done = False
        x = 0
        x_max = 0

        while not done:
            # each frame we run this code
            env.render()
            frames += 1
            ob = cv2.resize(ob, (res_x, res_y)) # fit to genesis screen
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY) # convert to 1d grayscale img
            ob = np.reshape(ob, (res_x, res_y))

            cv2.imshow('main', ob)
            cv2.waitKey(1)

            imgarray = np.ndarray.flatten(ob)

            nn_output = net.activate(imgarray)
            
            ob, reward, done, info = env.step(nn_output)

            current_fitness += reward
            
            if current_fitness > current_max_fitness:
                current_max_fitness = current_fitness
                idle_frames = 0
            else:
                idle_frames += 1

            # optional code for tracking the x position of sonic for fitness and the end of level state
            x = info['x']
            x_end = info['screen_x_end']
            
            if x > x_max:
                current_fitness += 1
                x_max = x
            
            if x == x_end and x > 500:
                current_fitness += 500000
                done = True
                
            done = True if idle_frames == 200 else done

            if done: 
                print(genome_id, current_fitness)
                
            genome.fitness = current_fitness
    

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-recurrentnetwork')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes, 5)

# plot figures
gens = range(len(stats.get_fitness_mean()))

plt.figure(figsize=(12,6))
plt.plot(gens, stats.get_fitness_mean(), label="Mean Fitness", linewidth=2)
plt.plot(gens, stats.get_fitness_stat(max), label="Best Fitness", linewidth=2)
plt.fill_between(gens, 
    np.array(stats.get_fitness_mean()) - np.array(stats.get_fitness_stdev()),
    np.array(stats.get_fitness_mean()) + np.array(stats.get_fitness_stdev()),
    color='gray',
    alpha=0.3,
    label="Std Dev"
)

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Population Fitness Over Generations")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# fitness heatmap
species_fitness = stats.get_species_fitness()
max_len = max(len(gen) for gen in species_fitness)
padded = []
for gen in species_fitness:
    row = list(gen) + [0] * (max_len - len(gen))
    padded.append(row)

data = np.array(padded)

plt.figure(figsize=(10,6))
plt.imshow(data, cmap='hot', aspect='auto')
plt.colorbar(label="Fitness")
plt.title("Fitness Heatmap (Species × Generation)")
plt.xlabel("Species Index")
plt.ylabel("Generation")
plt.show()

# visualize winner network and stats

visualize.plot_stats(stats, ylog=False, view=True, filename=r"C:\SonicNEAT\fitness_recurrentnetwork.png")
visualize.plot_species(stats, view=True, filename=r"C:\SonicNEAT\speciation_recurrentnetwork.png")
visualize.plot_spikes(winner, config, filename=r"C:\SonicNEAT\spikes_recurrentnetwork.gv")