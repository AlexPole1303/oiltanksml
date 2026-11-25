"""Compare radar repeatability in geometric volume units on identical tank samples."""
import math
from collections import defaultdict

from audit_shadow_scene_error import M3_PER_BBL
from build_s1_reference_matches import export, read

FIELDS = ("unaligned_height_difference_m", "aligned_height_difference_m", "scene_registered_height_difference_m")


def main():
    area = {}
    for row in read("umbra_sicd_close_pass_window_index"):
        tank = row["candidate_id"]
        gradient = math.pi * float(row["radius_m"]) ** 2 / M3_PER_BBL
        if area.setdefault(tank, gradient) != gradient:
            raise ValueError(f"Tank geometry differs across captures for {tank}")
    groups = defaultdict(list)
    for row in read("umbra_sicd_roof_arc_close_pairs"):
        if row["roof_type"] == "EFR" and row["scene_registered_height_difference_m"]:
            groups[row["first_product_id"], row["last_product_id"]].append(row)

    results = []
    for (first, last), paired in groups.items():
        accepted = [r for r in paired if r["scene_registered_both_pass"] == "True"]
        for label, rows in (("all_common_registration_inputs", paired), ("both_arc_diagnostics_pass", accepted)):
            if not rows:
                continue
            gradients = [area[r["candidate_id"]] for r in rows]
            estimates = {field: sum(float(r[field])*g for r, g in zip(rows, gradients, strict=True)) for field in FIELDS}
            results.append(dict(first_product_id=first, last_product_id=last,
                separation_seconds=float(rows[0]["separation_seconds"]), sample=label, tanks=len(rows),
                nominal_bbl_per_m=sum(gradients), no_registration_geometric_bbl=estimates[FIELDS[0]],
                local_registration_geometric_bbl=estimates[FIELDS[1]],
                scene_registration_geometric_bbl=estimates[FIELDS[2]],
                interpretation="Cylinder-equivalent difference between captures on matched EFR tanks. "
                "This is a repeatability diagnostic using the LiDAR radius. It does not use tank capacity charts or ground-truth inventory."))
    export("umbra_sicd_pair_volume_scale", results)
    for row in results:
        print(row["separation_seconds"], row["sample"], row["tanks"],
              *(round(row[f"{method}_registration_geometric_bbl"]) for method in ("no", "local", "scene")), flush=True)


if __name__ == "__main__":
    main()
