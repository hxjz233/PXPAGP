import numpy as np


def LoadResults(output_prefix, io_dict):
    E = None
    V = None
    S = None

    if io_dict["read_E"]:
        E = np.load(f"{output_prefix}_E.npy")
        print(f"Loaded E from {output_prefix}_E.npy, shape: {E.shape}")
    if io_dict["read_V"]:
        V = np.load(f"{output_prefix}_V.npy")
        print(f"Loaded V from {output_prefix}_V.npy, shape: {V.shape}")
    if io_dict["read_S"]:
        S = np.load(f"{output_prefix}_S.npy")
        print(f"Loaded S from {output_prefix}_S.npy, shape: {S.shape}")

    return E, V, S

def SaveResults(output_prefix, io_dict, E, V, S):
    if io_dict["write_E"] and E is not None:
        np.save(f"{output_prefix}_E", E)
        print(f"Saved E to {output_prefix}_E.npy, shape: {E.shape}")
    if io_dict["write_V"] and V is not None:
        np.save(f"{output_prefix}_V.npy", V)
        print(f"Saved V to {output_prefix}_V.npy, shape: {V.shape}")
    if io_dict["write_S"] and S is not None:
        np.save(f"{output_prefix}_S.npy", S)
        print(f"Saved S to {output_prefix}_S.npy, shape: {S.shape}")

def GetMeta(param):
    model = param['model']
    T = param['sym'][0]
    P = param['sym'][1]
    T_str = str(T) if T is not False else 'False'
    P_str = str(P) if P is not False else 'False'
    bound = param['bound']
    if 'sub' in param:
        sub = param['sub']
        return f"{model:s}_N={param['N']:d}_hxz={param['hxz']:.4f}_T={T_str:s}_P={P_str:s}_sub={sub:x}_bound={bound:s}"
    else:
        return f"{model:s}_N={param['N']:d}_hxz={param['hxz']:.4f}_T={T_str:s}_P={P_str:s}_bound={bound:s}"

def ParseFilename(filename):
    # Extract the parameters from the filename
    params = filename.split("_")
    N = int(params[1][2:])
    hxz = float(params[2][4:])
    T = int(params[3][2:]) if params[3][2:] != "False" else False
    P = int(params[4][2:]) if params[4][2:] != "False" else False

    gen_dict = dict(N=N, hxz=hxz, sym=(T, P))

    # Check if 'sub' is included in the filename
    if 'sub' in params[5]:
        sub = int(params[5][4:], 16)
        gen_dict['sub'] = sub

    return gen_dict