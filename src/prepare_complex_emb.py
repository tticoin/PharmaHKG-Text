import sys
import numpy as np


_, real_emb_in, complex_emb_out = sys.argv

real_emb = np.load(real_emb_in)

EMB_INIT_EPS = 2.0
gamma = 12.0
hidden_dim = 2 * real_emb.shape[1]
emb_init = (gamma + EMB_INIT_EPS) / hidden_dim
img_emb = np.random.uniform(emb_init, -emb_init, real_emb.shape)

complex_emb = np.concatenate((real_emb, img_emb), axis=1)
print(complex_emb.shape)
np.save(complex_emb_out, complex_emb)
