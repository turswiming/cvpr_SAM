import os
import json
path = "output/output_mask/"
outputfile = "decision.csv"
with open(outputfile, 'w') as out:
    for file in os.listdir(path):
        if file.endswith('.npy'):
            name_no_extension = os.path.splitext(file)[0]
            names = name_no_extension.split('_')
            names = names[0:4]
            name = "_".join(names)
            print(name)
            with open(path + name + "_bbox.json", 'r') as f:
                bbox = json.load(f)
                #{"floor": {
                # "min": [245.1200156377486, 227.16002174310844, 149.6287357659896],
                # "max": [290.3200156377486, 274.76002174310844, 150.92873576598961]},
                # "ceiling": {
                # "min": [245.1200156377486, 227.16002174310844, 152.39354358301333],
                # "max": [290.3200156377486, 274.76002174310844, 156.39354358301333]},
                # "x_size": 45.20000000000002,
                # "y_size": 47.599999999999994,
                # "z_size": 2.7648078170237227,
                # "point_number": 17517024,
                # "noise_rate": 0.25483535331115603}
            with open(path + name + "_room_data.json", 'r') as f:
                room = json.load(f)
                #{"room_number": 35, "large_room": 1, "small_room": 9}
            x_size = bbox["x_size"]
            y_size = bbox["y_size"]
            z_size = bbox["z_size"]
            point_number = bbox["point_number"]
            noise_rate = bbox["noise_rate"]
            room_number = room["room_number"]
            large_room = room["large_room"]
            small_room = room["small_room"]
            area = 2 * (x_size * y_size + x_size * z_size + y_size * z_size)
            out.write(name + ",")
            out.write(str(x_size) + ",")
            out.write(str(y_size) + ",")
            out.write(str(z_size) + ",")

            out.write(str(x_size/y_size) + ",")
            out.write(str(bbox["point_number"]/area) + ",")
            out.write(str(room_number) + ",")
            out.write(str(large_room) + ",")
            out.write(str(small_room) + ",")
            out.write(str(noise_rate) + "\n")

