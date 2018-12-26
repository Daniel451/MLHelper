import os
from collections import defaultdict


from .ImageLabelReader import DataObject


# data
# imagesets = ["bitbots-set00-01", "bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04", "bitbots-set00-05",
#              "bitbots-set00-06", "bitbots-set00-07", "bitbots-set00-08", "bitbots-set00-09", "bitbots-set00-10",
#              "bitbots-set00-11", "bitbots-set00-12", "bitbots-set00-13", "bitbots-set00-14", "bitbots-set00-15"]
# imagesets = ["bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04", "bitbots-set00-05"]
imagesets = ["test-wolves-01"]
# imagesets = ["2017_nagoya/sequences-jasper-euro-ball-1",
#              "2017_nagoya/sequences-jasper-kicking-euro-ball",
#              "2017_nagoya/sequences-misc-ball-robot-1",
#              "2017_nagoya/sequences-misc-ball-1",
#              "2017_nagoya/sequences-euro-ball-robot-1",
#              "2017_nagoya/euro-ball-game-1",
#              "test-nagoya-game-02"]
pathlist = [os.path.join(os.environ["ROBO_AI_DATA"], iset) for iset in imagesets]

data = DataObject(pathlist, batch_size=1, queue_size=16)

visited = defaultdict(int)

buffer = list()
for i in range(data.get_dataset_size() + 50):
    try:
        batch = data.get_next_batch()
        d = batch.get_labels()[0]
        print(f"[{i:0>5}/{data.get_dataset_size()}] [{len(buffer)} ERRORS]  checking '[{d['set']}/{d['file']}]'...")

        visited[f"{d['set']}/{d['file']}"] += 1
    except KeyError as e:
        print(e)
        buffer.append(e)

print()
print("### FINISHED ###")
print()

if len(buffer) > 0:
    print("exceptions")
    for m in buffer:
        print("key error", m)
else:
    print("loop ran without exceptions")

print()

print("checking if every image was visited")
print(f"entries in visited: {len(visited)}")
print(f"entries in dataset: {data.get_dataset_size()}")

print()
print("missing calls:")
diff = data.get_set_img().difference(visited.keys())
for e in diff:
    print(e)

print()
greater_one = {k: v for k, v in visited.items() if v > 1}

print("visited more than once")
for k, v in greater_one.items():
    print(k, v)

print()
print("datasets checked:")
for dset in imagesets:
    print(dset)
