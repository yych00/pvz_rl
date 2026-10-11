"""Generate an isolated training save for the supported game."""
import argparse
from pvz_gym.runtime.save_template import prepare_training_save

def main():
    parser = argparse.ArgumentParser(description="Prepare independent PvZ 1.2.0.1065 training saves")
    parser.add_argument("--game-exe")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        path = prepare_training_save(args.game_exe, args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print(f"Training template ready: {path} (PvZRL, 10 slots, adventure/shop cards)")

if __name__ == "__main__":
    main()
