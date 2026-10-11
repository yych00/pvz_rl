"""Manual high-frequency SeedChooser select/deselect stress probe."""

from __future__ import annotations

import sys
import time
from pathlib import Path

TOOLS_GYM = Path(__file__).resolve().parents[3]
PYTHON_DIR = TOOLS_GYM / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from pvz_gym.bridge.native import NativePvzBackend
from pvz_gym.bridge.seed_selection import SeedSelectionNativeBackend


def main():
    native = NativePvzBackend()
    b = SeedSelectionNativeBackend()

    print(f"Attach: {native._dll.pvz_gym_attach()}")
    print("\n[Step 1] Select level 2-1...")
    native._dll.pvz_gym_select_adventure_level(2, 1)
    for _ in range(10):
        time.sleep(0.4)
        if native._dll.pvz_gym_get_ui() == 2:
            break
    print(f"Current UI: {native._dll.pvz_gym_get_ui()}")

    print("\n[Step 2] High-frequency Select & Deselect stress test...")
    for round_idx in range(5):
        st = b.get_seed_selection()
        print(f"  Round {round_idx+1}: initial sel={st.selected_count}/{st.required_count}, seeds={st.selected_seeds}")
        mask = b.get_seed_selection_mask()
        picks = [i for i in range(64) if mask[i]][:4]
        for plant_type in picks:
            result = b.select_seed(plant_type)
            assert result.success, f"Failed to select plant {plant_type}: {result.message}"

        st_mid = b.get_seed_selection()
        print(f"    After selects: sel={st_mid.selected_count}/{st_mid.required_count}, seeds={st_mid.selected_seeds}")
        assert st_mid.selected_count == st.selected_count + len(picks), "Slot count mismatch"

        for plant_type in reversed(picks):
            result = b.deselect_seed(plant_type)
            assert result.success, f"Failed to deselect plant {plant_type}: {result.message}"

        st_end = b.get_seed_selection()
        print(f"    After deselects: sel={st_end.selected_count}/{st_end.required_count}, seeds={st_end.selected_seeds}")
        assert st_end.selected_count == st.selected_count, "Deselect slot count mismatch"

    print("\n>>> ALL 5 STRESS ROUNDS PASSED")

    print("\n[Step 3] Final Fill and Confirm to Battle...")
    st = b.get_seed_selection()
    mask = b.get_seed_selection_mask()
    avail = [i for i in range(64) if mask[i]]
    to_pick = st.required_count - st.selected_count
    for plant_type in avail[:to_pick]:
        b.select_seed(plant_type)

    st_full = b.get_seed_selection()
    print(f"Full deck: sel={st_full.selected_count}/{st_full.required_count}, can_confirm={st_full.can_confirm}")
    assert st_full.can_confirm, "Deck must be ready to confirm"

    b.confirm_seeds()
    for _ in range(10):
        time.sleep(0.5)
        ui = native._dll.pvz_gym_get_ui()
        if ui == 3:
            print("Successfully entered battle UI=3")
            break


if __name__ == "__main__":
    main()
