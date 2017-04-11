""" Automated model selection for those implemented in models directory

"""

import sys
import click
import random
import pandas as pd
from numpy import mean
from deap import base, creator
from deap import tools
from deap import algorithms

from models.maxout_new import Maxout

from feature_selection import get_l1_selection
from feature_selection import get_xgboost_selection


class ModelSelector():
    def __init__(self, df, features, eval_type, ngen):
        self.df = df
        self.features = features
        self.eval_type = eval_type
        self.ngen = ngen

    def evaluate(self, indi):
        """ Train model with individual parameters

        TODO: add eval_type to test best evaluation (log_loss, bayes_acc, etc.)
        TODO: add model_type to test best model (Maxout, Maxout Residual, etc.)
        """
        print indi
        model = Maxout(num_features=len(self.features),
                       num_layers=indi[0],
                       num_nodes=indi[1],
                       dropout_p=indi[2])
        scores = model.train_model(df=self.df,
                                  features=self.features,
                                  eval_type=self.eval_type)
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
        num_layer_min, num_layer_max = 1, 3
        num_nodes_min, num_nodes_max = 2, 100
        dropout_min, dropout_max = 0.0, 0.5

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0,-1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("num_layers", random.randint, num_layer_min, num_layer_max)
        toolbox.register("num_nodes", self.get_random_numnodes_for_poolsize, num_nodes_min, num_nodes_max)
        toolbox.register("dropout_p", random.uniform, dropout_min, dropout_max)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                    (toolbox.num_layers, toolbox.num_nodes, toolbox.dropout_p), n=1)

        toolbox.register("mate", tools.cxUniform)
        toolbox.register("mutate", self.random_mutation)
        toolbox.register("select", self.select_best)
        toolbox.register("evaluate", self.evaluate)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        return toolbox

    def select_best_model(self):
        """ """
        toolbox = self.get_toolbox()
        pop = toolbox.population(n=10)
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(self.ngen):
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
def run(ngen):
    for i, s in enumerate((2010,2011,2012,2013,2014,2015,2016)):
        if i == 0:
            df = pd.read_csv('data/final/{}_tourney_games.csv'.format(s))
            df['season'] = s
        else:
            tmp = pd.read_csv('data/final/{}_tourney_games.csv'.format(s))
            tmp['season'] = s
            df = df.append(tmp)

    features = df.keys().tolist()
    features.remove('won')
    features.remove('season')

    selector = ModelSelector(df=df, features=features,
                        eval_type='bayes_loss', ngen=ngen)
    selector.select_best_model()


if __name__ == "__main__":
    run()
