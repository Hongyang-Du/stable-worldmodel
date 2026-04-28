from pathlib import Path

import stable_worldmodel as swm

dataset_path = (
    Path(swm.data.utils.get_cache_dir())
    / 'datasets'
    / 'pointmaze-teleport-navigate-v0.h5'
)
output_path = (
    Path(swm.data.utils.get_cache_dir())
    / 'datasets'
    / 'pointmaze-teleport-navigate-v0-video'
)

swm.data.convert(dataset_path, output_path, dest_format='video')
