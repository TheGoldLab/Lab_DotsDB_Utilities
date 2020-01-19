import numpy as np
import h5py

filename = "test_vlen_datasets_np_bool.h5"

rows = [np.array([np.True_, np.False_]),
        np.array([np.True_, np.True_, np.False_])]

f = h5py.File(filename, 'x')  # create file, fails if exists

vlen_data_type = h5py.special_dtype(vlen=np.bool_)
dset = f.create_dataset("vlen_matrix", (2,),
                        compression="gzip",
                        compression_opts=9,
                        fletcher32=True,
                        dtype=vlen_data_type)


for r in range(len(rows)):
    dset[r] = rows[r]

f.flush()
f.close()

f = h5py.File(filename, 'r')
dsetr = f["vlen_matrix"]

for r in range(dsetr.shape[0]):
    print(dsetr[r])
