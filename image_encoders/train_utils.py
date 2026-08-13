import os
import pickle as pkl
import requests
from collections import defaultdict
from tqdm import tqdm

import imageio
import jax


def load_resnet10_params(agent, image_keys=("image",), public=True):
    """
    Load pretrained resnet10 params from github release to an agent.
    :return: agent with pretrained resnet10 params
    """
    file_name = "resnet10_params.pkl"
    if not public:  # if github repo is not public, load from local file
        with open(file_name, "rb") as f:
            encoder_params = pkl.load(f)
    else:  # when repo is released, download from url
        # Construct the full path to the file
        file_path = os.path.expanduser("~/.serl/")
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_path = os.path.join(file_path, file_name)
        # Check if the file exists
        if os.path.exists(file_path):
            print(f"The ResNet-10 weights already exist at '{file_path}'.")
        else:
            url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
            print(f"Downloading file from {url}")

            # Streaming download with progress bar
            try:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024  # 1 Kibibyte
                t = tqdm(total=total_size, unit="iB", unit_scale=True)
                with open(file_path, "wb") as f:
                    for data in response.iter_content(block_size):
                        t.update(len(data))
                        f.write(data)
                t.close()
                if total_size != 0 and t.n != total_size:
                    raise Exception(
                        "Error, something went wrong with the download")
            except Exception as e:
                raise RuntimeError(e)
            print("Download complete!")

        with open(file_path, "rb") as f:
            encoder_params = pkl.load(f)

    param_count = sum(x.size for x in jax.tree_leaves(encoder_params))
    print(
        f"Loaded {param_count/1e6}M parameters from ResNet-10 pretrained on ImageNet-1K"
    )

    new_params = agent.state.params

    # need to change the names to be 
    for image_key in image_keys:
        if "modules_calql_actor" in new_params.keys():
            new_encoder_params = new_params["modules_calql_actor"]["encoder"][
                f"encoder_{image_key}"
            ]
        else:
            if "modules_critic" not in new_params.keys():
                new_encoder_params = new_params["modules_actor"]["encoder"][
                    f"encoder_{image_key}"
                ]
            else:
                if "encoder" not in new_params["modules_critic"].keys():
                    new_encoder_params = new_params["modules_actor"]["encoder"][
                        f"encoder_{image_key}"
                    ]
                else:
                    new_encoder_params = new_params["modules_critic"]["encoder"][
                        f"encoder_{image_key}"
                    ]
            
        if "pretrained_encoder" in new_encoder_params:
            new_encoder_params = new_encoder_params["pretrained_encoder"]
        for k in new_encoder_params:
            if k in encoder_params:
                new_encoder_params[k] = encoder_params[k]
                print(f"replaced {k} in pretrained_encoder")

    agent = agent.replace(state=agent.state.replace(params=new_params))
    return agent
