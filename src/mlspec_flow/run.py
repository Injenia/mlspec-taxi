import sys, argparse
from configs.base_run import get_run
#from trainer.task import main as train
import logging

def parse_cli(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('step', type=str, choices=["initialize", "preprocess", "train", "select_best_model", "validate", "evaluate"])
    parser.add_argument('run', type=str)
    parser.add_argument('--force', default=False, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--local', default=False, action='store_true')
    return parser.parse_known_args(argv)
    
def main(step, run, force, debug, local, cli_params):
    run = get_run(run)
    if step == "initialize":
        run.initialize()
    elif step == "preprocess":
        run.preprocess(debug=debug, overwrite_previous=force)
    elif step == "train":
        if local:
            run.train(overwrite_previous=force, cli_params=cli_params)
        else:
            run.train_on_cloud(overwrite_previous=force, cli_params=cli_params)
    elif step == "select_best_model":
        run.select_best_model(overwrite_previous=force)
    elif step == "validate":
        run.validate(overwrite_previous=force)
    elif step == "evaluate":
        run.evaluate()
    logging.info("done")
    
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    known_args, unknown_args = parse_cli(sys.argv[1:])
    main(**vars(known_args), cli_params=unknown_args)