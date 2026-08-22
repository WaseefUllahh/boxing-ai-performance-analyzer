import sys
from pathlib import Path
from config import CFG
from src.result_manager import ResultManager

def main():
    rm = ResultManager("empty_test.mp4")
    rm.set_video_metadata(1920, 1080, 30.0, 10)
    
    # Completely empty data
    fight_stats = {"fighters": {}, "rounds": {}, "total_time_seconds": 0.0}
    strikes = []
    defenses = []
    movement = {}
    
    rm.export_results(fight_stats, strikes, defenses, movement)
    
    print(f"Exported empty data to {rm.output_dir}")

if __name__ == "__main__":
    main()
