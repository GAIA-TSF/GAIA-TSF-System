
import os
import json
import argparse


def load_index(root_dir):

    index_file = os.path.join(root_dir, 'experiments_index.json')

    if not os.path.exists(index_file):
        raise RuntimeError(
            f'No experiment index found: {index_file}'
        )

    with open(index_file) as f:
        return json.load(f)


def load_params(root_dir, experiment):

    params_file = os.path.join(
        root_dir,
        experiment,
        'params.json'
    )

    if not os.path.exists(params_file):
        return {}

    with open(params_file) as f:
        return json.load(f)


def build_rows(root_dir, experiments):

    rows = []

    for exp in experiments:

        params = load_params(root_dir, exp['experiment'])

        model = params.get('model', {})
        trainer = params.get('trainer', {})

        row = {
            'experiment': exp['experiment'],
            'loss': exp['best_test_loss'],
            'hidden_size': model.get('hidden_size', '-'),
            'layers': model.get('num_layers', '-'),
            'dropout': model.get('dropout', '-'),
            'lr': trainer.get('learning_rate', '-'),
        }

        rows.append(row)

    return rows


def print_table(rows):

    print()
    print('Experiment Comparison')
    print('-' * 75)

    header = (
        f'{'Experiment':15s}'
        f'{'Loss':10s}'
        f'{'Hidden':10s}'
        f'{'Layers':10s}'
        f'{'Dropout':10s}'
        f'{'LR':10s}'
    )

    print(header)
    print('-' * 75)

    for r in rows:

        print(
            f'{r['experiment']:15s}'
            f'{r['loss']:<10.4f}'
            f'{str(r['hidden_size']):10s}'
            f'{str(r['layers']):10s}'
            f'{str(r['dropout']):10s}'
            f'{str(r['lr']):10s}'
        )

    print('-' * 75)


def main():

    parser = argparse.ArgumentParser(
        description='Compare GAIA-TSF experiments'
    )

    parser.add_argument(
        '--root',
        type=str,
        default='tsf_projects',
        help='Experiment root directory',
    )

    args = parser.parse_args()

    experiments = load_index(args.root)

    experiments = sorted(
        experiments,
        key=lambda x: x['best_test_loss']
    )

    rows = build_rows(args.root, experiments)

    print_table(rows)


if __name__ == '__main__':
    main()
    