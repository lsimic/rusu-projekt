import bpy
import csv
import math
import os

# get path to .csv relative to .blend file location
filepath = bpy.data.filepath
directory = os.path.dirname(filepath)
csv_path = os.path.join(directory, "trained_outputs.csv")

current_frame = 1
car_obj = bpy.data.objects["car"]
target_obj = bpy.data.objects["target"]
with open(csv_path, newline="") as csvfile:
    output_reader = csv.reader(csvfile, delimiter=",")
    for row in output_reader:
        # set car rotation and location
        car_co = (float(row[2])*50, float(row[3])*50, 0.0)
        car_rot_z = math.atan2(float(row[5]), float(row[4]))
        bpy.context.scene.frame_set(current_frame)
        car_obj.select_set(True)
        bpy.context.view_layer.objects.active = car_obj
        car_obj.location = car_co
        bpy.ops.anim.keyframe_insert(type="Location")
        car_obj.rotation_euler[2] = car_rot_z
        bpy.ops.anim.keyframe_insert(type="Rotation")
        car_obj.select_set(False)
        # set target locations and rotations
        target_co = (float(row[0])*50, float(row[1])*50, 0.0)
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        target_obj.location = target_co
        bpy.ops.anim.keyframe_insert(type="Location")
        target_obj.select_set(False)
        # increment frame
        current_frame += 1
 