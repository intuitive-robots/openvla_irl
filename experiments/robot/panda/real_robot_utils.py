import os
import time

import imageio
import numpy as np
import tensorflow as tf
import torch

from experiments.robot.panda.real_robot_env import RREnvClient

# Initialize important constants and pretty-printing mode in NumPy.
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_real_robot_env(cfg, model=None):
    """Get Real Robot control environment."""
    # Set up the Real Robot client
    env = RREnvClient(
        name="OpenVLA Client",
        host=cfg.host_ip,
        port=cfg.port,
    )
    assert env.connect()
    return env


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_video(rollout_images, idx):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_data(rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx):
    """
    Saves rollout data from an episode.

    Args:
        rollout_orig_images (list): Original rollout images (before preprocessing).
        rollout_images (list): Preprocessed images.
        rollout_states (list): Proprioceptive states.
        rollout_actions (list): Predicted actions.
        idx (int): Episode index.
    """
    os.makedirs("./rollouts", exist_ok=True)
    path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.npz"
    # Convert lists to numpy arrays
    orig_images_array = np.array(rollout_orig_images)
    images_array = np.array(rollout_images)
    states_array = np.array(rollout_states)
    actions_array = np.array(rollout_actions)
    # Save to a single .npz file
    np.savez(path, orig_images=orig_images_array, images=images_array, states=states_array, actions=actions_array)
    print(f"Saved rollout data at path {path}")


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    obs["full_image"] = new_obs["full_image"]
    obs["image_primary"] = new_obs["image_primary"]
    obs["proprio"] = new_obs["proprio"]
    return obs
