import data.dataset as ds

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Standard SMPL kinematic tree (parent-child indices)
SMPL_EDGES = [
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), 
    (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), 
    (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)
]

def animate_smpl(data, edges, fps=20):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize lines for the skeleton
    lines = [ax.plot([], [], [], 'o-', lw=2)[0] for _ in range(len(edges))]
    
    # Set axis limits based on your data range
    ax.set_xlim3d([np.min(data[:,:,0]), np.max(data[:,:,0])])
    ax.set_ylim3d([np.min(data[:,:,1]), np.max(data[:,:,1])])
    ax.set_zlim3d([np.min(data[:,:,2]), np.max(data[:,:,2])])

    def update(frame):
        for i, (p1, p2) in enumerate(edges):
            x = [data[frame, p1, 0], data[frame, p2, 0]]
            y = [data[frame, p1, 1], data[frame, p2, 1]]
            z = [data[frame, p1, 2], data[frame, p2, 2]]
            lines[i].set_data(x, y)
            lines[i].set_3d_properties(z)
        return lines

    # interval = 1000 / fps
    ani = FuncAnimation(fig, update, frames=len(data), 
                        interval=1000/fps, blit=True)
    ani.save('animation.mp4', writer='ffmpeg', fps=20)
    
    plt.show()
    
def recover_from_hml3d(data, mean, std):
    # 1. Reverse Standard Scaling
    data = data * std + mean
    
    # 2. Extract components (indices based on standard HumanML3D format)
    # Root linear velocity is usually at indices [1, 2]
    r_velocity = data[:, 1:3]
    # Root height is usually at index 3
    r_height = data[:, 3]
    # Relative joint positions are usually at indices [4:67]
    j_pos_relative = data[:, 4:67].reshape(-1, 21, 3)
    
    # 3. Integrate Root Path
    # Compute the trajectory on the XZ plane
    root_xz = np.cumsum(r_velocity, axis=0) 
    
    # 4. Reconstruct Global Joints
    # Create an empty array for 22 joints (root + 21 others)
    joints = np.zeros((len(data), 22, 3))
    
    # Set the Root (Joint 0)
    joints[:, 0, 0] = root_xz[:, 0] # X
    joints[:, 0, 1] = r_height       # Y (Height)
    joints[:, 0, 2] = root_xz[:, 1] # Z
    
    # Set the other 21 joints by adding root position to relative positions
    joints[:, 1:, :] = j_pos_relative + joints[:, :1, :]
    
    return joints


if __name__ == "__main__":

    mean = np.load('./data/HumanML3D/Mean.npy')
    std  = np.load('./data/HumanML3D/Std.npy')

    data_root = "./data/HumanML3D"
    dataset = ds.HumanML3DDataset(data_root, split="train")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[1]

    print("Sample keys:", sample.keys())
    print("Motion shape:", sample["motion"].shape)
    print("Text:", sample["text"])
    print("Length:", sample["length"])
    print("ID:", sample["id"])

    joints = recover_from_hml3d(sample["motion"].numpy(), mean, std)
    print(joints.shape)
    animate_smpl(joints, SMPL_EDGES)
