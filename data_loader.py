from pathlib import Path
import json
from collections import defaultdict

DATA_ROOT = Path("DataSet/v1.0-mini")


def load_json(name):
    with open(DATA_ROOT / f"{name}.json") as f:
        return json.load(f)


def build_lookup(table):
    return {item['token']: item for item in table}


def extract_pedestrian_instances(sample_annotations, instances, categories):
    cat_lookup = build_lookup(categories)
    inst_lookup = build_lookup(instances)

    pedestrian_instances = set()

    for ann in sample_annotations:
        inst = inst_lookup[ann['instance_token']]
        category = cat_lookup[inst['category_token']]['name']

        if "pedestrian" in category:
            pedestrian_instances.add(ann['instance_token'])

    return pedestrian_instances


def build_trajectories(sample_annotations, pedestrian_instances):
    ann_lookup = build_lookup(sample_annotations)

    visited = set()
    trajectories = []

    for ann in sample_annotations:
        token = ann['token']

        if token in visited:
            continue

        if ann['instance_token'] not in pedestrian_instances:
            continue

        # go to start of chain
        current = ann
        while current['prev'] != "":
            current = ann_lookup[current['prev']]

        traj = []

        # traverse forward
        while current:
            visited.add(current['token'])

            x, y, _ = current['translation']
            traj.append([x, y])

            if current['next'] == "":
                break

            current = ann_lookup[current['next']]

        if len(traj) >= 10:
            trajectories.append(traj)

    return trajectories


def create_windows(trajectories):
    samples = []

    for traj in trajectories:
        for i in range(len(traj) - 9):
            window = traj[i:i+10]

            # normalize
            x0, y0 = window[0]
            window = [[x - x0, y - y0] for x, y in window]

            # compute velocity
            vel = []
            for j in range(len(window)):
                if j == 0:
                    vel.append([0, 0])
                else:
                    dx = window[j][0] - window[j-1][0]
                    dy = window[j][1] - window[j-1][1]
                    vel.append([dx, dy])

            obs = []
            for j in range(4):
                obs.append([
                    window[j][0],
                    window[j][1],
                    vel[j][0],
                    vel[j][1]
                ])

            future = window[4:10]

            samples.append((obs, future))

    return samples


def main():
    print("Loading data...")

    sample_annotations = load_json("sample_annotation")
    instances = load_json("instance")
    categories = load_json("category")

    print("Filtering pedestrians...")
    ped_instances = extract_pedestrian_instances(
        sample_annotations, instances, categories
    )

    print("Building trajectories...")
    trajectories = build_trajectories(sample_annotations, ped_instances)

    print("Creating training samples...")
    samples = create_windows(trajectories)

    print(f"Total trajectories: {len(trajectories)}")
    print(f"Total samples: {len(samples)}")

    # debug one sample
    obs, future = samples[0]
    print("OBS shape:", len(obs), len(obs[0]))
    print("FUTURE shape:", len(future), len(future[0]))


if __name__ == "__main__":
    main()