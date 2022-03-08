import matplotlib.pyplot as plt
import numpy as np


def remove_data():
    bad_words = ["{'depth_aruco_1':"]

    with open('logfile.log') as oldfile, open('updatelogfile1.log', 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in bad_words):
                newfile.write(line)


def read_log(filename):
    file = open(filename, "r")
    order = ['aruco_1_4', 'depth_aruco_1', 'dist_to_human_0',
             'depth_human_0', 'dist_to_aruco_1', 'dist_to_aruco_4']
    data = []

    for line in file.readlines():
        frame_elements = line[31:].split(",")
        details = [i.strip().replace("{", "").replace("}", "").replace(
            "'", "").split(":")[1].strip() for i in frame_elements]

        structure = {key: value for key, value in zip(order, details)}
        data.append(structure)
    return data


filename = "updatelogfile1.log"
data = read_log(filename)


y1 = []
y2 = []
for i, frame in enumerate(data):
    # print("i: {0} {1}".format(i, frame['dist_to_aruco_4']))
    y1.append(float(frame['dist_to_human_0']))
    y2.append(float(frame['dist_to_aruco_4']))


difference_array = np.subtract(y1, y2)
squared_array = np.square(difference_array)
mse = squared_array.mean()
print("MSE: {0}".format(mse))

x = np.linspace(0, len(data)-1, len(data))

plt.figure()
plt.plot(x, y1, color='red', label='Calculated distance')
plt.plot(x, y2, color='green', label='ArUco distance')
plt.legend()
# naming the x axis
plt.xlabel('x - Frame')
# naming the y axis
plt.ylabel('y - Distance to human (m)')
plt.title("Distance to human")
plt.savefig("Distance to human")

norm = np. linalg. norm(difference_array)
normal_error = difference_array/norm
plt.figure()
plt.plot(x, normal_error, color='blue', label='Difference')
plt.legend()
# naming the x axis
plt.xlabel('x - Frame')
# naming the y axis
plt.ylabel('y - Normal error')
plt.title("Error")
plt.savefig("Error")

plt.figure()
plt.plot(x, difference_array, color='blue', label='Difference')
plt.legend()
# naming the x axis
plt.xlabel('x - Frame')
# naming the y axis
plt.ylabel('y - Difference (m)')
plt.title("Difference\Error")
plt.savefig("Difference\Error")

y1_selected = []
y2_selected = []
for i, frame in enumerate(data):
    # print("i: {0} {1}".format(i, frame['dist_to_aruco_4']))
    if np.abs(float(frame['dist_to_human_0']) - float(frame['dist_to_aruco_4'])) < 0.30:
        y1_selected.append(float(frame['dist_to_human_0']))
        y2_selected.append(float(frame['dist_to_aruco_4']))
    # else:
    #     y1_selected.append(0.0)
    #     y2_selected.append(0.0)

x = np.linspace(0, len(y1_selected)-1, len(y1_selected))

plt.figure()
plt.plot(x, y1_selected, color='red', label='Calculated distance')
plt.plot(x, y2_selected, color='green', label='ArUco distance')
plt.legend()
# naming the x axis
plt.xlabel('x - Frame')
# naming the y axis
plt.ylabel('y - Distance to human (m)')
plt.title("Distance to human filtered")
plt.savefig("Distance to human filtered")

difference_array_selected = np.subtract(y1_selected, y2_selected)
squared_array_selected = np.square(difference_array_selected)
mse_selected = squared_array_selected.mean()
print("MSE after cleaning dataset: {0}".format(mse_selected))

plt.figure()
plt.plot(x, difference_array_selected, color='blue', label='Difference')
plt.legend()
# naming the x axis
plt.xlabel('x - Frame')
# naming the y axis
plt.ylabel('y - Difference (cm)')
plt.title("Difference\Error filtered")
plt.savefig("Difference\Error filtered")

plt.show()
