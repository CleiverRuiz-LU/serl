import time
import mujoco as mj
import mujoco.viewer as mjv
import numpy as np

# Robot joint positions
Q_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Home position (neutral pose)
Q_GOAL = np.array([0.3, -0.5, 0.2, -2.0, 0.1, 1.8, 1.0])          # Goal position

# Interactive controls
reset = False
KEY_SPACE = 32
KEY_R = 82


def generate_trajectory(q_start, q_end, num_steps=100):
    """Generate smooth joint trajectory using linear interpolation with smoothing"""
    trajectory = []
    for i in range(num_steps):
        # Use smooth interpolation (ease-in-out)
        t = i / (num_steps - 1)
        # Smooth step function for better motion
        smooth_t = 3 * t**2 - 2 * t**3
        q_interp = q_start + smooth_t * (q_end - q_start)
        trajectory.append(q_interp)
    return np.array(trajectory)

def key_callback(keycode):
    global reset
    if keycode == KEY_SPACE:
        reset = True
    elif keycode == KEY_R:
        reset = True

def main():
    """Main function to run the Panda robot movement test"""
    global reset
    
    # Load the MuJoCo model with Panda robot
    model = mj.MjModel.from_xml_path('../envs/xmls/arena.xml')
    data = mj.MjData(model)
    
    # Initialize robot at home position
    data.qpos[0:7] = Q_HOME
    mj.mj_forward(model, data)  # Forward kinematics to update state
    
    # Generate trajectory
    traj_q = generate_trajectory(Q_HOME, Q_GOAL, 100)
    
    print("Starting Panda robot movement test...")
    print("Controls:")
    print("  SPACE: Reset to start position")
    print("  R: Reset to start position")
    print("  Close viewer to exit")
    
    # Launch MuJoCo viewer with trajectory execution
    with mjv.launch_passive(model, data, key_callback=key_callback) as viewer:
        step = 0
        direction = 1  # 1 for forward, -1 for backward
        cycle_count = 0  # Track number of complete back-and-forth cycles
        max_cycles = 4  # Stop after 4 complete cycles
        movement_complete = False  # Flag to indicate when movement is done
        
        while viewer.is_running():
            if reset:
                # Reset to starting position
                step = 0
                direction = 1
                cycle_count = 0
                movement_complete = False
                data.qpos[0:7] = traj_q[0]
                reset = False
                print("Reset to starting position")
            elif not movement_complete:
                # Execute trajectory only if movement is not complete
                data.qpos[0:7] = traj_q[step]
                
                # Move to next step
                step += direction
                
                # Reverse direction at endpoints
                if step >= len(traj_q) - 1:
                    cycle_count += 1
                    print(f"Reached goal position - cycle {cycle_count} half completed")
                    
                    # Check if we've completed the required number of cycles
                    if cycle_count >= max_cycles:
                        movement_complete = True
                        print(f"Movement complete! Robot stopped at goal position after {max_cycles} cycles.")
                        # Keep robot at goal position
                        data.qpos[0:7] = traj_q[-1]
                    else:
                        direction = -1
                        step = len(traj_q) - 1
                        print("Reversing trajectory back to start")
                elif step <= 0:
                    direction = 1
                    step = 0
                    print("Returned to start position - moving to goal again")
            
            # Step the simulation only if not paused
            if not movement_complete:
                mj.mj_step(model, data)
            else:
                # When movement is complete, just update position without stepping
                # This pauses the simulation while keeping the robot at goal position
                pass
            
            # Sync viewer
            viewer.sync()
            
            # Control timing (50 Hz)
            time.sleep(0.02)
    
    print("Panda movement test completed.")

if __name__ == '__main__':
    main()