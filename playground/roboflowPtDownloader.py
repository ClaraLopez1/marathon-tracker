from inference import get_model

model = get_model(
    model_id="bib-detector/2",  # sin el "rbnr/"
    api_key="N337MrbrRDEqdpWVm0Hc"
)

import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.infer(dummy)
print("Listo!")