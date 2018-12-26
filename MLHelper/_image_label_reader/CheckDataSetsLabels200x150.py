import os
from collections import defaultdict


from .ImageLabelReader import DataObject


# data
imagesets = ["bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04", "bitbots-set00-05", "bitbots-set00-06",
             "bitbots-set00-07", "bitbots-set00-08", "bitbots-set00-09", "bitbots-set00-10", "bitbots-set00-11",
             "bitbots-set00-12", "bitbots-set00-13", "bitbots-set00-14"]
# imagesets = ["bitbots-set00-02", "bitbots-set00-03", "bitbots-set00-04", "bitbots-set00-05"]
# imagesets = ["bitbots-set00-02"]
pathlist = [os.path.join(os.environ["ROBO_AI_DATA"], iset) for iset in imagesets]

data = DataObject(pathlist, batch_size=1, queue_size=16, img_dim=(200, 150))

visited = defaultdict(int)

buffer = list()
for i in range(data.get_dataset_size() + 1):
    batch = data.get_next_batch()
    d = batch.get_labels()[0]
    print(f"[{i:0>5}/{data.get_dataset_size()}] [{len(buffer)} ERRORS]  checking '[{d['set']}/{d['file']}]'...")

    visited[f"{d['set']}/{d['file']}"] += 1

    if d["width"] <= 4 or d["height"] <= 4 or d["center_x"] == 0 or d["center_y"] == 0 \
            or d["center_x"] == 200 or d["center_y"] == 150:
        msg = "\n"
        msg += f"[{d['set']}/{d['file']}]\n"
        msg += f"width: {d['width']}, height: {d['height']}\n"
        msg += f"cx: {d['center_x']}, cy {d['center_y']}\n"
        msg += f"(x1/y1), (x2/y2): ({d['x1']}/{d['y1']}), ({d['x2']}/{d['y2']})\n"
        buffer.append(msg)

print()
print("### FINISHED ###")
print()

for m in buffer:
    print(m)

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
