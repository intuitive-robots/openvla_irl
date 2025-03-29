from typing import Dict, Tuple, Union
import cv2
import gymnasium as gym
import numpy as np
import zmq


def euler_to_quaternion(euler_angles):
    """
    Euler angles in radian follow the convention: [alpha, beta, gamma] -> R_z(gamma) @ R_y(beta) @ R_x(alpha)
    Quaternions follow the convention: x, y, z, w
    """

    half_roll = euler_angles[0] / 2.0
    half_pitch = euler_angles[1] / 2.0
    half_yaw = euler_angles[2] / 2.0

    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)

    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy

    return np.array([x, y, z, w])

class RREnvClient(gym.Env):

    """
    
    This class is a gym environment that communicates with an EnvServer to implement its methods.
    The outputs are adapted to work with Octo.

    Example Usage:

        from real_robot_env.env_client import EnvClient
        import numpy as np

        client = EnvClient(
            host = "127.0.0.1",
            port = 6060,
        )

        assert client.connect()

        obs, info = client.reset()
        obs, reward, term, trunc, info = client.step(np.array([0.2719, -0.5165, 0.2650, -1.6160, -0.0920, 1.6146, -1.7760, -1]))

        client.close()
    
    Launch Server with:

        TODO

    """

    def __init__(
            self,
            name = "Environment Client",
            host: str = "127.0.0.1",
            port: int = 6060,
            sticky_gripper_num_steps: int = 1,
        ):

        self.name = name
        self.host = host
        self.port = port
        self._addr = f"tcp://{self.host}:{self.port}"
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0
        self.observation_space = None
        # OpenVLA as well as WidowXEnv uses the format: (delta xyz, euler rotation angles, grasp action)
        # Euler angles in radian follow the convention: [alpha, beta, gamma] -> R_z(gamma) @ R_y(beta) @ R_x(alpha)
        self.action_space = gym.spaces.Box(
            low = np.zeros((7,)),
            high = np.ones((7,)),   
            dtype = np.float64,
        )

    def connect(self) -> bool:

        # Connect to server
        print(f"Connecting to {self.name}...")
        try:
            self._socket.connect(self._addr)
            print("Success")
        except Exception as e:
            print("Failed with exception: ", e)
            return False

        # Update observation space with actual image shapes
        reference_obs = self.get_observation()
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(  # Not used at the moment
                    low = np.zeros(reference_obs["image_primary"].shape),
                    high = 255 * np.ones(reference_obs["image_primary"].shape),
                    dtype = np.uint8,
                ),
                "full_image": gym.spaces.Box(
                    low = np.zeros(reference_obs["full_image"].shape),
                    high = 255 * np.ones(reference_obs["full_image"].shape),
                    dtype = np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low = np.ones((8,)) * -1,
                    high=np.ones((8,)),
                    dtype=np.float64,
                ),
            }
        )

        return True

    def close(self) -> bool:

        self._socket.disconnect(self._addr)
        print(f"Closed connection to {self.name}")
        return True

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, str]]:
            
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Convert action
        action = self._convert_action(action)

        # Sticky gripper logic
        if (action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = -1.0 if self.is_gripper_closed else 1.0

        # Send request
        self._send_step_request(action)  # TODO Add parameter for blocking
        
        # Receive response
        results = self._receive_results()

        # Convert observation
        obs = self._convert_obs(results["observation"])

        return obs, results["reward"], results["terminated"], results["truncated"], results["info"]

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:

        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_reset_request()
        
        # Receive response
        results = self._receive_results()

        # Convert observation
        obs = self._convert_obs(results["observation"])

        # Reset sticky gripper
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        return obs, results["info"]
    
    def get_observation(self) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
        
        if self._socket.closed:
            raise Exception(f"Not connected to {self.name}")
        
        # Send request
        self._send_get_observation_request()
        
        # Receive response
        results = self._receive_results()

        # Convert observations
        obs = self._convert_obs(results["observation"])

        return obs

    def _convert_obs(self, obs):

        # Assume that the image has the correct shape, otherwise this should be fixed in the server's gym environment
        image_obs = obs["primary_camera"]
        
        # Concatenate joint positions and gripper width
        proprio = np.concatenate([obs["joint_pos"], obs["gripper_width"]], axis=-1)
        
        return {
            "image_primary": image_obs,  # OpenVLA does not seem to use this
            "full_image": image_obs,  # Original code uses obs["full_image"], which should correspond to the camera output
            "proprio": proprio,
        }

    def _convert_action(self, action):
        
        # Decompose action
        ee_delta_xyz = action[:3]
        ee_euler_rotation = action[3:6]
        grasp_action = action[[6,]]

        # Convert Euler angles to a quaternion
        ee_quat_rotation = euler_to_quaternion(ee_euler_rotation)

        return np.concatenate((ee_delta_xyz, ee_quat_rotation, grasp_action), axis=0)

    def _send_step_request(self, action: np.ndarray):

        flags = 0

        # Determine action metadata
        action_metadata = {
            "dtype": str(action.dtype),
            "shape": action.shape,
        }
        
        # Send request and metadata
        request = {
            "command": "step",
            "action_metadata": action_metadata
        }
        self._socket.send_json(request, flags | zmq.SNDMORE)

        # Send action data
        self._socket.send(action, flags, copy=False, track=False)

    def _send_reset_request(self):

        flags = 0

        # Send request
        request = {"command": "reset"}
        self._socket.send_json(request, flags)

    def _send_get_observation_request(self):

        flags = 0

        # Send request
        request = {"command": "get_observation"}
        self._socket.send_json(request, flags)

    def _receive_results(self) -> Dict[str, Union[str, int, float, bool, np.ndarray, Dict[str, str]]]:

        flags = 0

        # Receive metadata and simple results
        results = self._socket.recv_json(flags)
        obs_metadata = results.pop("observation_metadata")

        # Receive observation
        data_parts = self._socket.recv_multipart(flags, copy=False, track=False)

        # Reconstruct observation
        obs = {}
        for (key, metadata), data in zip(obs_metadata.items(), data_parts):
            buf = memoryview(data)
            array = np.frombuffer(buf, dtype=metadata["dtype"]).reshape(metadata["shape"])
            obs[key] = array
        results["observation"] = obs

        return results
