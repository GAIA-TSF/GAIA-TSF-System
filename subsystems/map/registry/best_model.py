import os
import json
import argparse


def load_index(root_dir):
    index_file = os.path.join(root_dir, 'experiments_index.json')

    if not os.path.exists(index_file):
        raise RuntimeError(f'No experiment index found: {index_file}')

    with open(index_file) as f:
        return json.load(f)


def find_best_experiment(experiments):
    if len(experiments) == 0:
        raise RuntimeError('No experiments registered')

    best = min(experiments, key=lambda x: x['best_test_loss'])

    return best


def main():
    parser = argparse.ArgumentParser(description='Return best trained model')

    parser.add_argument(
        '--root', type=str, default='tsf_experiments', help='Experiment root directory'
    )

    parser.add_argument(
        '--print-path', action='store_true', help='Print full model path'
    )

    args = parser.parse_args()

    experiments = load_index(args.root)

    best = find_best_experiment(experiments)

    print()
    print('Best experiment')
    print('----------------------------')
    print('Name:', best['experiment'])
    print('Best test loss:', best['best_test_loss'])

    if args.print_path:
        model_path = os.path.join(args.root, best['experiment'], 'best_model.pt')

        print('Model path:', model_path)


if __name__ == '__main__':
    main()
