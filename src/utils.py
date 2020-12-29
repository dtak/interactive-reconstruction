import os
import json
import numpy as np

# Add file extensions to TensorflowJS weight protobufs for more universal web
# server compatibility
def rename_tensorflowjs_manifests(json_path):
    with open(json_path, 'r') as f:
        manifest = json.loads(f.read())

    path = json_path.replace('/decoder_web/weights_manifest.json', '')

    manifest2 = []
    for m in manifest:
        paths2 = []
        for p in m['paths']:
            paths2.append(p + '.pb')
            weights = os.path.join(path, 'decoder_web/' + p)
            os.system(f"mv {weights} {weights}.pb")
        m['paths'] = paths2
        manifest2.append(m)

    with open(json_path, 'w') as f:
        f.write(json.dumps(manifest2))

# Fix JSON encoding for Python objects containing numpy arrays
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
