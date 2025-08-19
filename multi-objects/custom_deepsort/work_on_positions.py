import json 

# tr_path = "run_3/trajectories.json"
tr_path = "line_car_1/trajectories.json"
tr_path_2 = "line_car_1/trajectories_2.json"

def calculate_relative_line_heghit(frame_width, frame_height):
    start_x = int(frame_width * (2/8))
    start_y = int(frame_height * (5/8))
    end_x = int(frame_width * (12/16))
    end_y = int(frame_height * (5/8))
    # return the line height 
    return start_y

# function to get the relative height of the line based on the drawn line 
def calculate_line_height(frame_width, frame_height, is_straight= True, start_x_ratio=None, start_y_ratio=None, end_x_ratio=None, end_y_ratio=None):
    """
    Parameters
    ---------

    start_x_ratio: 1/4
    start_y_ratio:3/8
    end_x_ratio:
    end_y_ratio: 5/8
    """
    start_point_x = int(frame_width * (start_x_ratio))
    start_point_y = int(frame_height * (start_y_ratio))
    end_point_x = frame_width
    end_point_y = int(frame_height * (end_y_ratio))
    if is_straight:
        return start_point_y
    else:
        # Compute the slope (m) and intercept (b) of the line equation y = mx + b
        m = (end_point_y - start_point_y) / (end_point_x - start_point_x)  # Slope
        b = start_point_y - m * start_point_x  # Intercept 
        return m, b

# def calculate_line_height(frame_width, frame_height, start_x_ratio=None, start_y_ratio=None, end_x_ratio=None, end_y_ratio=None):
#     """
#     Parameters
#     ---------

#     start_x_ratio: 1/4
#     start_y_ratio:3/8
#     end_x_ratio:
#     end_y_ratio: 5/8
#     """
#     start_point_x = int(frame_width * (start_x_ratio))
#     start_point_y = int(frame_height * (start_y_ratio))
#     end_point_x = frame_width * end_x_ratio
#     end_point_y = int(frame_height * (end_y_ratio))
#     return start_point_x, start_point_y, end_point_x, end_point_y


def side_of_line(px, py, x1, y1, x2, y2):
    """
    Determines which side of the line a point (px, py) is on.
    Parameter
    --------
    px, px: (coord)
        given point which we want to check
    x1, y1: (coord)
        position of starting point of the line 
    x2, y2: (coord)
        position of end point of the line 
    Returns:
    - Negative value if below the line
    - Positive value if above the line
    - Zero if exactly on the line
    """
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)


# get line height based on the frame size
height = 800
width = 1280
line_y = calculate_relative_line_heghit(width, height)
print(f"Height of the line with respect to frame: {line_y}")

# Open and read the JSON file
with open(tr_path, "r") as file:
    data = json.load(file) 

with open(tr_path_2, "r") as file:
    data_2 = json.load(file) 

# print the data 
# print(data)

# list for incoming and outgoing car id's
incoming_car = []
outgoing_car = [] 

# loooping through the data
# for obj_id, positions in data.items():
#     print(f"Object ID: {obj_id}")
#     for time_step, (x, y) in enumerate(positions):
#         if y > line_y:
#             if obj_id not in incoming_car:
#                 incoming_car.append(obj_id)
#                 print(f"Car {obj_id} passed in incoming way")
#         else:
#             if obj_id not in outgoing_car:
#                 outgoing_car.append(obj_id)
#                 print(f"Car {obj_id} passed in the outgoing way")
#         print(f"  Time {time_step}: x={x}, y={y}")

for obj_id, positions in data.items():
    for i in range(1, len(positions)):
        prev_x, prev_y = positions[i - 1]
        curr_x, curr_y = positions[i]

        if prev_y < line_y and curr_y >= line_y:
            print(f"Car {obj_id} crossed the line (y={line_y}) moving DOWN -> Incoming.")
            incoming_car.append(obj_id)
        elif prev_y > line_y and curr_y <= line_y:
            print(f"Car {obj_id} crossed the line (y={line_y}) moving UP -> Outgoing.")
            outgoing_car.append(obj_id)


# print the incoming and outgoing cars 
car_passing = {
    'incoming': incoming_car,
    'outgoing': outgoing_car
}
print(f"Incoming and outgoing report:\n {car_passing}")

############ Non-Straight Line ################
incoming_car_2 = []
outgoing_car_2 = [] 


# Now calculate for the non-straight line 
line_slope, line_intercept = calculate_line_height(width, height, is_straight= False, start_x_ratio=2/8, 
                                                   start_y_ratio=5/8, end_x_ratio=12/16, end_y_ratio=7/8)

for obj_id, positions in data_2.items():
    for i in range(1, len(positions)):
        prev_x, prev_y = positions[i - 1]
        curr_x, curr_y = positions[i]

        # Get the y-values of the crossing line at prev_x and curr_x
        # m-> slope, b-> intercept
        expected_prev_y = line_slope * prev_x + line_intercept
        expected_curr_y = line_slope * curr_x + line_intercept

        # Check if the object moved from one side of the line to the other
        if prev_y < expected_prev_y and curr_y >= expected_curr_y:
            print(f"Object {obj_id} crossed the line UPWARD -> Incoming..")
            incoming_car_2.append(obj_id)
        elif prev_y > expected_prev_y and curr_y <= expected_curr_y:
            print(f"Object {obj_id} crossed the line DOWNWARD -> Outgoing ..")
            outgoing_car_2.append(obj_id)

# x1,y1,x2,y2 = calculate_line_height(width, height, start_x_ratio=2/8, 
#                                                    start_y_ratio=5/8, end_x_ratio=12/16, end_y_ratio=7/8)

# for obj_id, trajectory in data_2.items():
#         if len(trajectory) < 2:
#             continue  # Need at least two points to detect movement

#         prev_x, prev_y = trajectory[-2]  # Previous position
#         curr_x, curr_y = trajectory[-1]  # Current position

#         prev_side = side_of_line(prev_x, prev_y, x1,y1,x2,y2)
#         print(f"previous side: {prev_side}")
#         curr_side = side_of_line(curr_x, curr_y, x1,y1,x2,y2)
#         print(f"current side: {curr_side}")

#         # Detect crossing by checking if the sign changed (i.e., object moved to the other side)
#         if prev_side < 0 and curr_side >= 0:
#             print(f"Object {obj_id} is INCOMING.")
#             incoming_car_2.append(obj_id)
#         elif prev_side > 0 and curr_side <= 0:
#             print(f"Object {obj_id} is OUTGOING.")
#             outgoing_car_2.append(obj_id)

# print the incoming and outgoing cars 
car_passing_2 = {
    'incoming': incoming_car_2,
    'outgoing': outgoing_car_2
}
print(f"Incoming and outgoing report Approach-2:\n {car_passing_2}")

# ID specific positions 
# obj_id = "1"
# if obj_id in data:
#     positions = data[obj_id]  # Get positions for object 1
#     print(f"Trajectory of Object {obj_id}: {positions}")


dd = {
    'incoming': [],
    'outgoing': []
}

print(f"Initial empty dict: {dd}")

dd['incoming'].append(23)

print(f"After inserting values: {dd}")