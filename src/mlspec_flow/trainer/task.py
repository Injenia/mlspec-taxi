import sys, argparse
from configs.base_run import get_run

def parse_cli(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=str)
    parser.add_argument('--force', default=False, action='store_true')
    return parser.parse_known_args(argv)
    
def main(run, force, cli_params):
    run = get_run(run)
    run.train(overwrite_previous = force, cli_params=cli_params)
    
if __name__ == "__main__":
    #logging.getLogger().setLevel(logging.INFO)
    known_args, unknown_args = parse_cli(sys.argv[1:])
    main(**vars(known_args), cli_params=unknown_args)