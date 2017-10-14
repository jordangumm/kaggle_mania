from pyflow import WorkflowRunner
import os


class ModelSelector(WorkflowRunner):
    def workflow(self):
        self.addTask('model_selector', nCores=4, command=['python', 'model_selection.py',
                      '/home/ubuntu/kaggle_mania/kaggle_mania/march_mania/data/final/base_games.csv'])

if __name__ == "__main__":
    model_selector = ModelSelector()
    model_selector.run(mode='local', nCores=4,
                       dataDirRoot=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'workflow_runner'))
