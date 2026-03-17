
import os
import json
import argparse


def load_index(root_dir):

    index_file = os.path.join(root_dir, 'experiments_index.json')

    if not os.path.exists(index_file):
        print()
        print('No experiment index found.')
        print('Run a training experiment first.')
        print(f'Expected location: {index_file}')
        return []

    with open(index_file) as f:
        return json.load(f)


def print_table(experiments):

    print()
    print('Registered Experiments')
    print('-' * 50)
    print(f'{'Experiment':20s} {'Best Test Loss':15s}')
    print('-' * 50)

    for exp in experiments:
        print(
            f'{exp['experiment']:20s} '
            f'{exp['best_test_loss']:.4f}'
        )

    print('-' * 50)


def main():

    parser = argparse.ArgumentParser(
        description='List GAIA-TSF experiments'
    )

    parser.add_argument(
        '--root',
        type=str,
        default='tsf_experiments',
        help='Experiment root directory',
    )

    args = parser.parse_args()

    experiments = load_index(args.root)

    if len(experiments) == 0:
        return        

    experiments = sorted(
        experiments,
        key=lambda x: x['best_test_loss']
    )

    print_table(experiments)


if __name__ == '__main__':
    main()
