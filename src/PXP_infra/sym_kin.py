from quspin.basis.user import (
    next_state_sig_32,
    pre_check_state_sig_32,
    op_sig_32,
    map_sig_32,
)  # user_basis dtypes
from numba import carray, cfunc
from numba import uint32, int32  # numba data types
import numpy as np


######  function to call when applying operators
@cfunc(op_sig_32, locals=dict(s=int32, b=uint32))
def op(op_struct_ptr, op_str, ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    ind = N - ind - 1  # convention for QuSpin for mapping from bits to sites.
    s = (((op_struct.state >> ind) & 1) << 1) - 1
    b = 1 << ind
    #
    if op_str == 120:  # "x" is integer value 120 (check with ord("x"))
        op_struct.state ^= b
    elif op_str == 121:  # "y" is integer value 120 (check with ord("y"))
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j * s
    elif op_str == 122:  # "z" is integer value 120 (check with ord("z"))
        op_struct.matrix_ele *= s
    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err


#
op_args = np.array([], dtype=np.uint32)

# define particle conservation and op dicts
op_dict = dict(op=op, op_args=op_args)

#
######  function to filter states/project states out of the basis
#

@cfunc(
    pre_check_state_sig_32,
    locals=dict()
)
def testfunc(s, N, args):
    # positions = []
    # for i in range(N):
    #     if (s >> i) & 1:
    #         positions.append(i)
    
    # if not positions:
    #     return False
    # return positions[0]
    return args[0]


@cfunc(
    pre_check_state_sig_32,
    locals=dict(count=uint32),
)
def count_ups(s, N, args):
    """works for all system sizes N."""
    count = 0
    for i in range(N):
        if (s >> i) & 1:
            count += 1
    return count == args[2]


@cfunc(
    pre_check_state_sig_32,
    locals=dict(switch_count=uint32)
)
def count_switches(s, N, args):
    """
    Determines the number of transitions from odd to even positions in the binary representation of the state.
    
    Parameters:
    state (int): The state represented as an integer.
    N (int): The number of sites in the lattice.
    
    Returns:
    int: The switch count.
    """
    positions = []
    for i in range(N):
        if (s >> i) & 1:
            positions.append(i)
    
    if not positions:
        return args[1] == 0
    
    # Count switches from odd to even positions
    switch_count = 0
    for j in range(1, len(positions)):
        if positions[j-1] % 2 != positions[j] % 2:
            switch_count += 1
    
    # Check the circular switch
    if positions[-1] % 2 != positions[0] % 2:
        switch_count += 1
    
    switch_count /= 2
    
    # Adjust switch count based on conditions
    if switch_count == 0:
        if positions[0] % 2 == 0:       # 0b0101
            switch_count = 0xFFFFFFFF - 2
        else:
            switch_count = 0xFFFFFFFF - 1
    
    return args[1] == switch_count

@cfunc(
    pre_check_state_sig_32,
    locals=dict(s_shift_left=uint32, s_shift_right=uint32),
)
def pre_check_state(s, N, args):
    """imposes that that a bit with 1 must be preceded and followed by 0,
    i.e. a particle on a given site must have empty neighboring sites.
    #
    Works only for lattices of up to N=32 sites (otherwise, change mask)
    #
    """
    mask = 0xFFFFFFFF >> (32 - N)  # works for lattices of up to 32 sites
    if args[0] == 1:    # is PBC
        # cycle bits left by 1 periodically
        s_shift_left = ((s << 1) & mask) | ((s >> (N - 1)) & mask)
        #
        # cycle bits right by 1 periodically
        s_shift_right = ((s >> 1) & mask) | ((s << (N - 1)) & mask)
        #
        
        # if args[1] == 0xFFFFFFFF:
        #     return (((s_shift_right | s_shift_left) & s)) == 0
        # else:
        #     return (((s_shift_right | s_shift_left) & s)) == 0 and count_switches(s, N, args)
    else:
        # cycle bits left by 1 periodically
        s_shift_left = (s << 1) & mask
        #
        # cycle bits right by 1 periodically
        s_shift_right = (s >> 1) & mask
        #
        # return (((s_shift_right | s_shift_left) & s)) == 0

    flag = (((s_shift_right | s_shift_left) & s)) == 0
    if args[0] == 1 and args[1] != 0xFFFFFFFF:
        flag = flag and count_switches(s, N, args)
    if args[2] != 0xFFFFFFFF:
        flag = flag and count_ups(s, N, args)

    return flag
    
#

# define pre_check_state
pre_check_state_tup = (
    pre_check_state
)  # None gives a null pinter to args


#
######  define symmetry maps
#
@cfunc(
    map_sig_32,
    locals=dict(
        shift=uint32,
        xmax=uint32,
        x1=uint32,
        x2=uint32,
        period=int32,
        l=int32,
    ),
)
def translation(x, N, sign_ptr, args):
    """works for all system sizes N."""
    shift = args[0]  # translate state by shift sites
    period = N  # periodicity/cyclicity of translation
    xmax = args[1]
    #
    l = (shift + period) % period
    x1 = x >> (period - l)
    x2 = (x << l) & xmax
    #
    return x2 | x1


#
@cfunc(
    map_sig_32,
    locals=dict(
        out=uint32,
        s=int32,
    ),
)
def parity(x, N, sign_ptr, args):
    """works for all system sizes N."""
    out = 0
    s = args[0]  # N-1
    #
    out ^= x & 1
    x >>= 1
    while x:
        out <<= 1
        out ^= x & 1
        x >>= 1
        s -= 1
    #
    out <<= s
    return out