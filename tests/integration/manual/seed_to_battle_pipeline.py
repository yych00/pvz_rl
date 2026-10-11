"""Manual real-game probe for SeedChooser -> Battle transition."""

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
    cur_ui = native._dll.pvz_gym_get_ui()
    print(f"Current UI: {cur_ui}")

    if cur_ui != 2:
        print("\n[Step 1] Selecting level 2-1 (level_id=11) to enter Seed Selection...")
        code = native._dll.pvz_gym_select_adventure_level(2, 1)
        print(f"  select_level result: {code}")
        for _ in range(10):
            time.sleep(0.5)
            cur_ui = native._dll.pvz_gym_get_ui()
            if cur_ui == 2:
                break
        print(f"  Entered UI: {cur_ui}")

    st = b.get_seed_selection()
    print(f"\n[Step 2] Current Seed Selection: present={st.present}, req={st.required_count}, sel={st.selected_count}/{st.required_count}")
    print(f"  Current selected: {st.selected_seeds}")

    mask = b.get_seed_selection_mask()
    avail = [i for i in range(64) if mask[i]]
    print(f"  Available seeds ({len(avail)}): {avail[:10]}")

    to_pick = st.required_count - st.selected_count
    for picked, plant_type in enumerate(avail):
        if picked >= to_pick:
            break
        result = b.select_seed(plant_type)
        print(f"  Selected plant {plant_type}: success={result.success}")
        time.sleep(0.1)

    st2 = b.get_seed_selection()
    print(f"\n[Step 3] After filling: sel={st2.selected_count}/{st2.required_count}, can_confirm={st2.can_confirm}, seeds={st2.selected_seeds}")

    if st2.can_confirm:
        print("\n[Step 4] Calling confirm_seeds()...")
        t0 = time.perf_counter()
        result = b.confirm_seeds()
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"  confirm_seeds: success={result.success}, code={result.error_code}, msg='{result.message}' (took {dt:.2f}ms)")

        print("\n[Step 5] Monitoring UI transition...")
        for i in range(10):
            time.sleep(0.5)
            ui = native._dll.pvz_gym_get_ui()
            print(f"  t+{i*0.5:.1f}s: UI={ui}")
            if ui == 3:
                print("\n>>> SUCCESS: Seed Selection transitioned to Battle UI (UI=3)")
                break


if __name__ == "__main__":
    main()
