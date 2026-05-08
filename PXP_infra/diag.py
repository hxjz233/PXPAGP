from quspin.operators import hamiltonian
from quspin.basis.user import user_basis  # Hilbert space user basis
import numpy as np

from sym_kin import translation, parity, op_dict, pre_check_state_tup


def GetBasis(params):
    N = params['N']

    t_param, p_param = params['sym']
    isPBC = 1 if params['bound'] == "PBC" else 0
    maps = {}
    T_args = np.array([1, (1 << N) - 1], dtype=np.uint32)
    P_args = np.array([N - 1], dtype=np.uint32)
    
    if isPBC and t_param is not False:
        maps[f"T_block"] = (translation, N, t_param, T_args)
    if p_param is not False:
        maps[f"P_block"] = (parity, 2, p_param, P_args)

    sub = params.get('sub', 0xFFFFFFFF)
    Nup = params.get('Nup', 0xFFFFFFFF)
    pre_check_state = (pre_check_state_tup, np.array([isPBC, sub, Nup], dtype=np.uint32))

    return user_basis(
        np.uint32,
        N,
        op_dict,
        allowed_ops=set("xyz"),
        sps=2,
        pre_check_state=pre_check_state,
        Ns_block_est=300000,
        **maps,
    )

def Diag(params, basis, calc_V):
    H = GetHam(params, basis)
    if calc_V:
        E, V = H.eigh()
    else:
        E = H.eigvalsh()
        V = None

    if calc_V:
        return E, V
    else:
        return E

def GetHam(params, basis):
    N = params['N']
    hxz = params['hxz']
    t_param, p_param = params['sym']
    isPBC = 1 if params['bound'] == "PBC" else 0

    X_list = [[1.0, i] for i in range(N)]
    if isPBC:
        XZ_list = [[hxz, i, (i + 2) % N] for i in range(N)]
        ZX_list = [[hxz, (i - 2) % N, i] for i in range(N)]
    else:
        XZ_list = [[hxz, i, i + 2] for i in range(N - 2)]
        ZX_list = [[hxz, i - 2, i] for i in range(2, N)]

    static = [
        ["x", X_list], ["zx", ZX_list], ["xz", XZ_list]
    ]
    # compute Hamiltonian, no checks have been implemented
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
    if not isPBC or t_param is False or t_param == 0:
        H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
    else:
        H = hamiltonian(static, [], basis=basis, dtype=np.complex64, **no_checks)
    # make elements modulus smaller than 1e-6 zero
    # H = H.tocsr()
    # # H.eliminate_zeros()
    # Harray = H.toarray()
    # print(H)
    # # print(Harray-Harray.T)
    # print(np.where(np.abs(Harray-Harray.T)>1e-6))
    return H