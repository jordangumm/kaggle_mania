""" Automated model selection for those implemented in models directory

"""

import sys
import click
import random
import pandas as pd
import numpy as np
from numpy import mean
from deap import base, creator
from deap import tools
from deap import algorithms

import bagging_procedure


class ModelSelector():
    def __init__(self, train_df, test_df, features, eval_type, ngen, model_type):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.eval_type = eval_type
        self.ngen = ngen
        self.model_type = model_type

    def evaluate(self, indi):
        """ Train model with individual parameters

        TODO: add eval_type to test best evaluation (log_loss, bayes_acc, etc.)
        TODO: add model_type to test best model (Maxout, Maxout Residual, etc.)
        """
        num_layers = indi[0]
        num_nodes = indi[1]
        dropout_p = indi[2]
        weight_decay = indi[3]
        eta = indi[4]

        print 'num_layers:\t{}'.format(num_layers)
        print 'num_nodes:\t{}'.format(num_nodes)
        print 'dropout_p:\t{}'.format(dropout_p)
        print 'weight_decay:\t{}'.format(weight_decay)
        print 'learning rate:\t{}'.format(eta)

        scores = bagging_procedure.train_with_bagging(train_df=self.train_df,
            features=self.features, verbose=False, batch_size=1, num_epochs=99999,
            num_layers=num_layers,num_nodes=num_nodes,dropout_p=dropout_p,learning_rate=eta,
            early_stop_rounds=10, num_baggs=3, weight_decay=weight_decay)

        print tuple(scores)
        return tuple(scores)

    def get_random_numnodes_for_poolsize(self, low, high, poolsize=2):
        """ Return integer divisible by pool size

        Number of nodes must be divisible by pool size (2)
        Half of number of nodes must also be divisible by pool size (2)
        TODO: pool size as hyperparameter??? maxout paper stated poolsize 2 is great
        """
        num = random.randint(low, high)
        while (float(num)/float(poolsize)) % poolsize or num % poolsize:
            num = random.randint(low, high)
        return num

    def random_mutation(self, mutant, toolbox):
        """ TODO """
        print 'original: {}'.format(mutant)
        mutant = toolbox.population(n=10)[0]
        print 'mutant: {}'.format(mutant)
        sys.exit()
        return mutant

    def select_best(self, individuals, k):
        """ Return k best non-duplicate individuals """

        best = []
        best_added = 0
        while True:
            scores = [mean(ind.fitness.values) for ind in individuals]
            index = scores.index(min(scores))
            if individuals[index] not in best:
                best.append(individuals[index])
                best_added += 1
            print '{}: {}'.format(individuals[index], scores[index])
            del individuals[index]
            if best_added == k:
                break
        return best

    def get_toolbox(self):
        """

        eta: learning rate
        """
        num_layer_min, num_layer_max = 1, 2
        num_nodes_min, num_nodes_max = 2, 100
        dropout_min, dropout_max = 0.1, 0.5
        decay_min, decay_max = 1e-4, 1e-1
        eta_min, eta_max = 1e-4, 1e-1

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0)) # loss, bayes loss
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("num_layers", random.randint, num_layer_min, num_layer_max)
        toolbox.register("num_nodes", self.get_random_numnodes_for_poolsize, num_nodes_min, num_nodes_max)
        toolbox.register("dropout_p", random.uniform, dropout_min, dropout_max)
        toolbox.register("weight_decay", random.uniform, decay_min, decay_max)
        toolbox.register("eta", random.uniform, eta_min, eta_max)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.num_layers, toolbox.num_nodes, toolbox.dropout_p,
                                    toolbox.weight_decay, toolbox.eta), n=1)

        toolbox.register("mate", tools.cxUniform)
        toolbox.register("mutate", self.random_mutation)
        toolbox.register("select", self.select_best)
        toolbox.register("evaluate", self.evaluate)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    def select_best_model(self):
        """ """
        toolbox = self.get_toolbox()
        pop = toolbox.population(n=2)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self.ngen):
            print 'GENERATION {}'.format(g)
            offspring = toolbox.select(pop, 2)
            offspring = map(toolbox.clone, offspring)

            children = map(toolbox.clone, offspring)
            children = toolbox.mate(children[0], children[1], .5)

            new = toolbox.population(n=6)
            [new.append(off) for off in offspring]
            [new.append(ch) for ch in children]
            offspring = map(toolbox.clone, new)

            fitnesses = toolbox.map(toolbox.evaluate, offspring)
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
        best = toolbox.select(pop, 1)[0]
        print 'best: {}'.format(mean(best.fitness.values))


@click.command()
@click.argument('ngen', type=click.INT)
@click.argument('season_to_predict', type=click.INT)
@click.option('-model_type', default='maxout')
def run(ngen, season_to_predict, model_type):
    if season_to_predict not in xrange(2008,2017):
        sys.exit("season {} not in 2008-2016 prediction range".format(season_to_predict))

    for i, s in enumerate(xrange(2003,2017)):
        if i == 0:
            df = pd.read_csv('data/games/{}_tourney_diff_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('data/games/{}_tourney_diff_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    df = df.fillna(0.0)

    features = df.keys().tolist()
    features.remove('season')
    #features.remove('team_name')
    features.remove('won')

    """ Features removed due to LIME inspection """
    features.remove('seed')
    #features.remove('_seed')

    def normalize(data):
        for key in data.keys():
            if not key in features: continue
            mean = data[key].mean()
            std = data[key].std()
            data.loc[:, key] = data[key].apply(lambda x: x - mean / std)
        return data

    train_df = normalize(df[df['season'] < season_to_predict])
    test_df = normalize(df[df['season'] == season_to_predict])

    selector = ModelSelector(train_df=train_df, test_df=test_df, features=features,
                                                    eval_type='bayes_loss', ngen=ngen,
                                                    model_type=model_type)
    selector.select_best_model()


if __name__ == "__main__":
    run()
