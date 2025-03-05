from absl import app, flags

from experiments import experiment

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment_name", None, "Name of the experiment")
flags.DEFINE_boolean("debug", False, "Debug mode flag")
flags.DEFINE_boolean("tune", False, "Tune mode flag")


def main(*args, **kwargs):
    print(f"Starting experiment {FLAGS.experiment_name}, Tuning: {FLAGS.tune}")
    experiment.Experiment(
        config={
            "debug": FLAGS.debug,
            "experiment_name": FLAGS.experiment_name,
            "tune": FLAGS.tune,
        }
    ).run()


if __name__ == "__main__":
    app.run(main)
