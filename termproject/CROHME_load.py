import numpy as np
crohme = np.load('crohme_refined.npz')
np.savez_compressed('crohme_compressed.npz', x = crohme['x'], y = crohme['y'])