import numpy as np

def generate_toe_motion(mocap_data_len, push_off_start=0.35, push_off_end=0.5, max_plantarflexion=1.0):
    toe_motion = np.zeros((mocap_data_len, 2))  # [right_toe, left_toe]
    for i in range(mocap_data_len):
        phase = i / (mocap_data_len - 1)
        if phase < push_off_start:
            # Neutral before push-off
            toe_motion[i] = [0.0, 0.0]
        elif push_off_start <= phase <= push_off_end:
            # Linearly interpolate to max plantarflexion
            t = (phase - push_off_start) / (push_off_end - push_off_start)
            angle = t * max_plantarflexion
            toe_motion[i] = [angle, angle]
        elif push_off_end < phase < 0.9:
            # Interpolate back to neutral
            t = (phase - push_off_end) / (0.8 - push_off_end)
            angle = max_plantarflexion * (1 - t)
            toe_motion[i] = [angle, angle]
        else:
            # Neutral during landing and post-jump
            toe_motion[i] = [0.0, 0.0]
    return toe_motion

def append_toe_motion(mocap_data, toe_motion):
    for i in range(len(mocap_data.data_config)):
        mocap_data.data_config[i] = np.insert(mocap_data.data_config[i], 28, toe_motion[i][0])
        mocap_data.data_config[i] = np.insert(mocap_data.data_config[i], 36, toe_motion[i][1])
        # print(len(mocap_data.data_vel[i]))
        mocap_data.data_vel[i] = np.insert(mocap_data.data_vel[i], 27, mocap_data.data_vel[i][26])
        mocap_data.data_vel[i] = np.insert(mocap_data.data_vel[i], 35, mocap_data.data_vel[i][34])
    return mocap_data

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    else:
        return obj
    
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from utils.mocapTransform import MocapDM
    import json

    motion = MocapDM()
    motion.load_mocap("motions/humanoid3d_jump.txt")
    toe_motion = generate_toe_motion(len(motion.dataset))
    motion = append_toe_motion(motion, toe_motion)
    with open("motions/humanoid3d_jump_with_toes.txt", "w") as f:
        # print(type(motion.data_config))
        json.dump(convert_ndarray_to_list(motion.data_config), f)
    with open("motions/humanoid3d_jump_with_toes_vel.txt", "w") as f:
        # print(type(motion.data_config))
        json.dump(convert_ndarray_to_list(motion.data_vel), f)