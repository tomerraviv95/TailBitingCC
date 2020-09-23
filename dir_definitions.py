import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CODE_DIR = os.path.join(ROOT_DIR, 'python_code')
RESOURCES_DIR = os.path.join(ROOT_DIR, 'resources')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
WEIGHTS_DIR = os.path.join(RESULTS_DIR, 'weights')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

CONFIG_PATH = os.path.join(CODE_DIR, 'config.yaml')
BCH_MAT_PATH = os.path.join(RESOURCES_DIR, 'BCH_matrices')
TBCC_MAT_PATH = os.path.join(RESOURCES_DIR, 'TBCC_matrices')
LTE_TBCC_MAT_PATH = os.path.join(RESOURCES_DIR, 'LTE_TBCC_matrices')
